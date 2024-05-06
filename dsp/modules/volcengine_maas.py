"""Module for interacting with VolcEngine MaaS(Ark) Models."""
import os
from typing import Any, Dict, Optional
from typing_extensions import Self

import backoff
from pydantic_core import PydanticCustomError

from dsp.modules.lm import LM

try:
    import volcengine  # type: ignore[import-untyped]
    from volcengine.maas.v2 import MaasService
    from volcengine.maas import MaasException, ChatRole
except ImportError:
    pass


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
          f"calling function {details['target']} with kwargs "
          f"{details['kwargs']}")


def giveup_hdlr(details):
    """wrapper function that decides when to give up on retry"""
    if "rate limits" in details.message:
        return False
    return True

class VolcEngineMaaS(LM):
    """Wrapper around VolcEngineMaaS's API.

    Use endpoint to specify the model and regions.
    """

    def __init__(
        self,
        endpoint_id: str,
        api_ak: Optional[str] = None,
        api_sk: Optional[str] = None,
        api_domain: Optional[str] = None,
        api_region: Optional[str] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        endpoint_id : str
            endpoint_id specifies the model to use.
            Which endpoint_id from VolcEngine MaaS to use?
            doc is https://www.volcengine.com/docs/82379/1182403
        api_ak and api_sk : str
            ak and sk will be provided by the platform.
        **kwargs: dict
            Additional arguments to pass to the API provider, filled in parameters.
        """
        super().__init__(endpoint_id)
        self._init_maas(api_ak, api_sk, api_domain, api_region)
        self.endpoint_id = endpoint_id
        self.provider = "volcengine_maas"

        self.history: list[dict[str, Any]] = []
        self.kwargs = {
            "max_new_tokens": 1000,
            "min_new_tokens": 1,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0,
            "max_prompt_tokens": 32768,
        }
        self.kwargs.update(kwargs)

    def _init_maas(
        self,
        api_ak: Optional[str] = None,
        api_sk: Optional[str] = None,
        api_domain: Optional[str] = None,
        api_region: Optional[str] = None
    ) -> None:
        api_domain = api_domain or 'maas-api.ml-platform-cn-beijing.volces.com'
        api_region = api_region or 'cn-beijing'
        api_ak = api_ak or os.environ.get('VOLC_ACCESSKEY')
        api_sk = api_sk or os.environ.get('VOLC_SECRETKEY')
        self.maas = MaasService(api_domain, api_region)
        self.maas.set_ak(api_ak)
        self.maas.set_sk(api_sk)
        return

    def _prepare_params(
        self,
        parameters: Any,
    ) -> dict:
        params_mapping = {'max_tokens':'max_output_tokens'}
        params = {params_mapping.get(k, k): v for k, v in parameters.items()}
        params = {**self.kwargs, **params}
        # disallows "n" arguments
        n = params.pop("n", None)
        if n is not None and n > 1 and params['temperature'] == 0.0:
            params['temperature'] = 0.7
        return {k: params[k] for k in set(params.keys() & set(self.kwargs.keys()))}

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        params = self._prepare_params(raw_kwargs)
        req = {
            "parameters": params,
            "messages": [
                {
                    "role": ChatRole.USER,
                    "content": prompt
                },
            ]
        }
        try:
            response = self.maas.chat(self.endpoint_id, req)
        except MaasException as e:
            raise e
        #
        history = {
            "prompt": prompt,
            "response": {
                "prompt": prompt,
                "choices": [response.choice.message.content],
            },
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return [response.choice.message.content]

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Google whilst handling API errors"""
        return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ):
        return self.request(prompt, **kwargs)
