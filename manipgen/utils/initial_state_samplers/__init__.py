from manipgen.utils.initial_state_samplers.pick_cube_sampler import (
    FrankaPickCubePoseSampler,
)
from manipgen.utils.initial_state_samplers.place_cube_sampler import (
    FrankaPlaceCubePoseSampler,
)
from manipgen.utils.initial_state_samplers.pick_sampler import (
    FrankaPickPoseSampler,
)
from manipgen.utils.initial_state_samplers.place_sampler import (
    FrankaPlacePoseSampler,
)
from manipgen.utils.initial_state_samplers.grasp_handle_sampler import (
    FrankaGraspHandlePoseSampler,
)
from manipgen.utils.initial_state_samplers.open_sampler import (
    FrankaOpenPoseSampler,
)

samplers = {
    "pick_cube": FrankaPickCubePoseSampler,
    "place_cube": FrankaPlaceCubePoseSampler,
    "pick": FrankaPickPoseSampler,
    "place": FrankaPlacePoseSampler,
    "grasp_handle": FrankaGraspHandlePoseSampler,
    "open": FrankaOpenPoseSampler,
    "close": FrankaOpenPoseSampler,
}
