from manipgen.envs.franka_pick_cube import FrankaPickCube
from manipgen.envs.franka_place_cube import FrankaPlaceCube
from manipgen.envs.franka_pick import FrankaPick
from manipgen.envs.franka_place import FrankaPlace
from manipgen.envs.franka_grasp_handle import FrankaGraspHandle
from manipgen.envs.franka_open import FrankaOpen
from manipgen.envs.franka_close import FrankaClose
from manipgen.envs.franka_reach import FrankaReach

environments = {}

environments["pick_cube"] = FrankaPickCube
environments["place_cube"] = FrankaPlaceCube
environments["pick"] = FrankaPick
environments["place"] = FrankaPlace
environments["grasp_handle"] = FrankaGraspHandle
environments["open"] = FrankaOpen
environments["close"] = FrankaClose
environments['reach'] = FrankaReach
