from dataclasses import dataclass


@dataclass(frozen=True)
class LBEParams:
    # grid frame
    nx: int = 420
    ny: int = 180

    # obstacle
    cx: float = nx // 4
    cy: float = ny // 2
    r: float = ny // 9

    # environment
    Re: float = 1000.0
    uLB: float = 0.1
    nuLB: float = uLB * r / Re
    omega: float = 1 / (3 * nuLB + 0.5)



