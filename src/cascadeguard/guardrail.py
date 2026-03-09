import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Union

from .preranker import Preranker
from .fineranker import Fineranker, FinerankResponseStatus


@dataclass
class RankData:
    idx              : int
    prompt           : str
    output           : str
    sanitized_output : str   = None
    is_valid         : bool  = None
    risk_score       : float = None

    def __post_init__(self):
        # Convert numpy types to native Python types for better serialization
        if isinstance(self.is_valid, (np.bool_, bool)):
            object.__setattr__(self, 'is_valid', bool(self.is_valid))
        if isinstance(self.risk_score, (np.floating, np.integer)):
            object.__setattr__(self, 'risk_score', float(self.risk_score))

    def get_pair(self):
        return dict(prompt=self.prompt, output=self.output)

    def get_pair_prettystr(self):
        return f"Prompt: {self.prompt}\nOutput: {self.output}"

class CascadeGuard:

    def __init__(
        self,
        preranker: Union[dict, Preranker, None] = None,
        fineranker: Union[dict, Fineranker, None] = None,
    ):
        if preranker is None and fineranker is None:
            raise ValueError("At least one of preranker or fineranker must be provided")

        self.preranker = (
            preranker if isinstance(preranker, (Preranker, type(None)))
            else Preranker(**preranker)
        )
        if self.preranker:
            self.preranker = self.preranker.scanner

        self.fineranker = (
            fineranker if isinstance(fineranker, (Fineranker, type(None)))
            else Fineranker(**fineranker)
        )

    def apply(
        self,
        datas: List[Tuple[str, str]],
        winnow_down: bool = True,
        return_as_dict: bool = False,
    ):
        datas = self.apply_as_datas(datas, winnow_down=winnow_down)
        datas = self.apply_prerank(datas, winnow_down=winnow_down)
        datas = self.apply_finerank(datas)

        if return_as_dict:
            return self.apply_datas_as_listdict(datas)
        return datas

    def apply_as_datas(
        self,
        pairs: List[Tuple[str, str]],
        winnow_down: bool = True,
        return_as_dict: bool = False,
    ):
        datas = [
            RankData(
                idx=i,
                prompt=prompt,
                output=output,
                is_valid=winnow_down,
            )
            for i, (prompt, output) in enumerate(pairs)
        ]

        if return_as_dict:
            return self.apply_datas_as_listdict(datas)
        return datas

    def apply_datas_as_listdict(self, datas:List[RankData]):
        return [asdict(data) for data in datas]

    def apply_prerank(
        self,
        datas: Union[List[Tuple[str, str]], List[RankData]],
        winnow_down: bool = True,
        return_as_dict: bool = False,
    ):
        if datas and isinstance(datas[0], tuple):
            datas = self.apply_as_datas(datas, winnow_down=winnow_down)

        if self.preranker:
            for data in datas:
                data.sanitized_output, data.is_valid, data.risk_score = self.preranker.scan(data.prompt, data.output)

            datas = [d for d in datas if d.is_valid == winnow_down]

        if return_as_dict:
            return self.apply_datas_as_listdict(datas)
        return datas

    def apply_finerank(
        self,
        datas: List[RankData],
        winnow_down: bool = True,
        return_as_dict: bool = False,
    ):
        if datas and isinstance(datas[0], tuple):
            datas = self.apply_as_datas(datas, winnow_down=winnow_down)

        if self.fineranker:
            for data in datas:
                finerank_response = self.fineranker.rank(data.get_pair_prettystr())
                if finerank_response.status == FinerankResponseStatus.SUCCESS:
                    data.is_valid = finerank_response.is_valid

            datas = [d for d in datas if d.is_valid == winnow_down]

        if return_as_dict:
            return self.apply_datas_as_listdict(datas)
        return datas