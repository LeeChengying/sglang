import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Dict, Tuple
from enum import Enum, auto
from ucm.store.factory import UcmConnectorFactory
from ucm.store.ucmstore import Task
import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)

MHA_CACHE_NUMS = 2
MLA_CACHE_NUMS = 1

EXIST_FLAG_STR = "EXIST"
EXIST_FLAG = -1


@dataclass
class UnifiedCacheStoreConfig:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @staticmethod
    def load_config(storage_config: HiCacheStorageConfig, mem_pool_host: HostKVCache) -> "UnifiedCacheStoreConfig":
        try:
            kvc = storage_config.extra_config.get("kv_connector_extra_config")
            is_mla_model = storage_config.is_mla_model
            total_tp_size = storage_config.tp_size
            tp_rank = storage_config.tp_rank

            page_size = mem_pool_host.page_size
            element_size = mem_pool_host.get_size_per_token()
            layer_num = mem_pool_host.device_pool.layer_num

            ucm_cfg = kvc.get("ucm_connector_config")
            name = kvc.get("ucm_connector_name")

            cfg = dict(ucm_cfg)
            cfg["storage_backends"] = ucm_cfg.get("storage_backends")
            cfg["device"] = tp_rank
            cfg["role"] = "worker"
            cfg_base = page_size * element_size
            cfg["kv_block_size"] = (
                cfg_base * (1 if is_mla_model else total_tp_size)
            )
            cfg["io_size"] = cfg_base //layer_num
            return UnifiedCacheStoreConfig(
                name=name,
                config=cfg
            )
        except Exception as e:
            logger.error(f"Error loading UnifiedCacheStoreConfig: {e}")
            raise


class AttentionBackend(Enum):
    MHA = ("mha", MHA_CACHE_NUMS)
    MLA = ("mla", MLA_CACHE_NUMS)

    def __init__(self, label: str, cache_nums: int):
        self.label = label
        self.cache_nums = cache_nums


def _class_name(o) -> str:
    return o.__class__.__name__


class BaseBackendAdapter:
    create_func: Callable[[List[str]], List[int]]
    wait_func: Callable[["Task"], Any]
    lookup_func: Callable[[List[str]], List[bool]]
    commit_func: Callable[[List[str], bool], None]

    def __init__(
        self,
        backend: AttentionBackend,
        tp_size: int,
        local_rank: int,
        mem_pool_host,
    ):
        self.backend = backend
        self.tp_size = tp_size
        self.local_rank = local_rank
        self.mem_pool_host = mem_pool_host

    def build_transfer_data(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        is_set: bool
    ) -> Tuple[List[str], List[int], List[int], List[int]]:
        raise NotImplementedError

    def _get_cache_nums(self) -> int:
        return self.backend.cache_nums

    def commit_tasks(self, keys: List[str], is_success: bool) -> None:
        self.__class__.commit_func(keys, is_success)

    def _get_page_keys(self, keys: List[str]) -> List[str]:
        return [key for key in keys for _ in range(self._get_cache_nums())]

    def lookup_page_keys(self, keys: List[str]) -> int:
        lookup_results = self.__class__.lookup_func(keys)
        for i in range(len(lookup_results)):
            if lookup_results[i] != True:
                return i

        return len(lookup_results)

    def wait_tasks(self, tasks: List[Task]) -> List[bool]:
        raise NotImplementedError

    def get_dump_list(self, dump_key_list: List[str]) -> List[str]:
        raise NotImplementedError

    def _get_page_offsets(self, keys: List[str], elem_size: int) -> List[int]:
        raise NotImplementedError


class MhaBackendAdapter(BaseBackendAdapter):
    def wait_tasks(self, tasks: List[Task]) -> List[bool]:
        success_flags = []
        for i in range(0, len(tasks), 2):
            k_task = tasks[i]
            v_task = tasks[i + 1]

            success = self.__class__.wait_func(k_task) == 0 and \
            self.__class__.wait_func(v_task) == 0

            success_flags.append(success)
        return success_flags

    def build_transfer_data(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        is_set: bool
    ) -> Tuple[List[str], List[int], List[int], List[int]]:
        ptr_list, elem_size_list = self.mem_pool_host.get_page_buffer_meta(
            host_indices)
        elem_size = elem_size_list[0]
        key_list = self._get_page_keys(keys)
        offset_list = self._get_page_offsets(keys, elem_size)
        if is_set:
            lookup_results = self.__class__.lookup_func(keys)
        try:
            assert len(key_list) == len(ptr_list), \
            f"Key/Ptr list mismatch: {len(key_list)} vs {len(ptr_list)}"
            assert len(offset_list) == len(ptr_list), \
            f"Offset/Ptr list mismatch: {len(offset_list)} vs {len(ptr_list)}"
        except Exception as e:
            logger.error(f"Error in build_transfer_data: {e}")

        if is_set:
            create_key_list: List[str] = []
            new_key_list: List[str] = []
            new_offset_list: List[int] = []
            new_ptr_list: List[int] = []
            new_elem_size_list: List[int] = []

            for i in range(len(keys)):
                if lookup_results[i] != 1:
                    create_key_list.append(keys[i])
                    for k in range(self._get_cache_nums()):
                        new_key_list.append(key_list[2 * i + k])
                        new_offset_list.append(offset_list[2 * i + k])
                        new_ptr_list.append(ptr_list[2 * i + k])
                        new_elem_size_list.append(elem_size_list[2 * i + k])
                else:
                    for _ in range(self._get_cache_nums()):
                        new_key_list.append(EXIST_FLAG_STR)
                        new_offset_list.append(EXIST_FLAG)
                        new_ptr_list.append(EXIST_FLAG)
                        new_elem_size_list.append(EXIST_FLAG)

            self.__class__.create_func(create_key_list)
            return new_key_list, new_offset_list, new_ptr_list, new_elem_size_list

        return key_list, offset_list, ptr_list, elem_size_list

    def get_dump_list(self, dump_key_list: List[str]) -> List[str]:
        half_dump_len = len(dump_key_list) // 2
        for i in range(half_dump_len):
            assert dump_key_list[2 * i] == dump_key_list[2 * i + 1], \
            "dump key list generation error"
        return [dump_key_list[2 * i] for i in range(half_dump_len)]

    def _get_page_offsets(self, keys: List[str], elem_size: int) -> List[int]:
        offset_list: List[int] = []
        v_offset = self.tp_size * elem_size
        for _ in keys:
            offset_list.append(self.local_rank * elem_size)
            offset_list.append(self.local_rank * elem_size + v_offset)
        return offset_list


class MlaBackendAdapter(BaseBackendAdapter):
    def wait_tasks(self, tasks: List[Task]) -> List[bool]:
        return [self.__class__.wait_func(task) == 0 for task in tasks]

    def build_transfer_data(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        is_set: bool
    ) -> Tuple[List[str], List[int], List[int], List[int]]:
        ptr_list, elem_size_list = self.mem_pool_host.get_page_buffer_meta(
            host_indices)
        elem_size = elem_size_list[0]
        key_list = self._get_page_keys(keys)
        offset_list = self._get_page_offsets(keys, elem_size)
        if is_set:
            lookup_results = self.__class__.lookup_func(keys)
        try:
            assert len(key_list) == len(ptr_list), \
            f"Key/Ptr list mismatch: {len(key_list)} vs {len(ptr_list)}"
            assert len(offset_list) == len(ptr_list), \
            f"Offset/Ptr list mismatch: {len(offset_list)} vs {len(ptr_list)}"
            assert len(key_list) == len(keys), \
            f"MLA: Keys/Key list mismatch: {len(keys)} vs {len(key_list)}"
        except Exception as e:
            logger.error(f"Error in build_transfer_data: {e}")

        if is_set:
            create_key_list: List[str] = []
            new_key_list: List[str] = []
            new_offset_list: List[int] = []
            new_ptr_list: List[int] = []
            new_elem_size_list: List[int] = []

            for i in range(len(key_list)):
                if lookup_results[i] != 1:
                    create_key_list.append(key_list[i])

                    new_key_list.append(key_list[i])
                    new_offset_list.append(offset_list[i])
                    new_ptr_list.append(ptr_list[i])
                    new_elem_size_list.append(elem_size_list[i])
                else:
                    new_key_list.append(EXIST_FLAG_STR)
                    new_offset_list.append(EXIST_FLAG)
                    new_ptr_list.append(EXIST_FLAG)
                    new_elem_size_list.append(EXIST_FLAG)

            self.__class__.create_func(create_key_list)
            return new_key_list, new_offset_list, new_ptr_list, new_elem_size_list

        return key_list, offset_list, ptr_list, elem_size_list

    def get_dump_list(self, dump_key_list: List[str]) -> List[str]:
        return dump_key_list

    def _get_page_offsets(self, keys: List[str], elem_size: int) -> List[int]:
        return [0] * len(keys)


class UnifiedCacheStore(HiCacheStorage):

    def __init__(self, storage_config: HiCacheStorageConfig = None, mem_pool_host: HostKVCache = None):

        try:
            assert mem_pool_host is not None, "mem_pool_host cannot be None"
            ucm_store_config = UnifiedCacheStoreConfig.load_config(
                storage_config, mem_pool_host)
            self.store = UcmConnectorFactory.create_connector(
                ucm_store_config.name, ucm_store_config.config)
            self.mem_pool_host = mem_pool_host
            self.is_mla_model = storage_config.is_mla_model
            self.total_tp_size = storage_config.tp_size
            self.tp_size = storage_config.tp_size
            self.tp_rank = storage_config.tp_rank
            self.backend = self._init_backend_adapter(mem_pool_host)

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

    def _init_backend_adapter(self, mem_pool_host: HostKVCache):
        cls_name = _class_name(mem_pool_host)

        HOSTCACHE_TO_BACKEND = {
            "MHATokenToKVPoolHost": AttentionBackend.MHA,
            "MLATokenToKVPoolHost": AttentionBackend.MLA,
        }

        HOSTCACHE_TO_ADAPTER = {
            "MHATokenToKVPoolHost": MhaBackendAdapter,
            "MLATokenToKVPoolHost": MlaBackendAdapter,
        }

        if cls_name not in HOSTCACHE_TO_BACKEND:
            raise TypeError(
                f"Unsupported HostKVCache Type: {cls_name}. "
                f"Expected one of {list(HOSTCACHE_TO_BACKEND.keys())}"
            )

        self.backend = HOSTCACHE_TO_BACKEND[cls_name]
        self.adapter = HOSTCACHE_TO_ADAPTER[cls_name](
            backend=self.backend,
            tp_size=self.tp_size,
            local_rank=self.tp_rank,
            mem_pool_host=self.mem_pool_host,
        )

        cls = self.adapter.__class__
        cls.create_func = self.store.create
        cls.wait_func = self.store.wait
        cls.lookup_func = self.store.lookup
        cls.commit_func = self.store.commit

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        key_list, offset_list, ptr_list, elem_size_list = self.adapter.build_transfer_data(
            keys, host_indices, is_set=False)
        tasks = []
        for i in range(len(key_list)):
            tasks.append(self.store.fetch_data([key_list[i]], [offset_list[i]], [ptr_list[i]], [elem_size_list[i]]))

        return self.adapter.wait_tasks(tasks)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        key_list, offset_list, ptr_list, elem_size_list = self.adapter.build_transfer_data(
            keys, host_indices, is_set=True)
        dump_key_list = []
        tasks = []
        for i in range(len(key_list)):
            if key_list[i] != EXIST_FLAG_STR:
                dump_key_list.append(key_list[i])
                tasks.append(self.store.dump_data([key_list[i]], [offset_list[i]], [ptr_list[i]], [elem_size_list[i]]))

        dump_key_list = self.adapter.get_dump_list(dump_key_list)
        success_flags = self.adapter.wait_tasks(tasks)
        if self.tp_rank == 0:
            self.adapter.commit_tasks(dump_key_list, all(success_flags))

        return success_flags

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        exist_result = self.batch_exists([key])
        return exist_result[0] == 1

    def batch_exists(self, keys: List[str]) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        return self.adapter.lookup_page_keys(keys)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None
