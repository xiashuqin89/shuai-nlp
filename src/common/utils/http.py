import abc
from typing import Any, Optional, Dict

_token = {}


class Api:
    """
    Api interface
    """
    __slots__ = ('api_name', )

    def __init__(self,
                 api_name: Optional[str],
                 api_root: Optional[str],
                 *args,
                 **kwargs):
        self.api_name = api_name
        self._api_root = api_root.rstrip('/') if api_root else None

    @abc.abstractmethod
    def _get_access_token(self) -> Optional[Dict[str, Any]]:
        """
        Get access_token
        """
        pass

    @abc.abstractmethod
    def _is_token_available(self) -> bool:
        """
        Judge the access_token available
        -time
        -filed
        """
        pass

    @abc.abstractmethod
    def _handle_api_result(result: Optional[Dict[str, Any]]) -> Any:
        """
        Deal response's result
        """
        pass

    @abc.abstractmethod
    def call_action(self, action: str, **params) -> Optional[Dict[str, Any]]:
        """
        Send API request to call the specified action.
        """
        pass
