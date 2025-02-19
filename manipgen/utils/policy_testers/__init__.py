from manipgen.utils.policy_testers.base_tester import BaseTester
from manipgen.utils.policy_testers.reach_test import FrankaReachTester
from manipgen.utils.policy_testers.pick_cube_tester import FrankaPickCubeTester
from manipgen.utils.policy_testers.place_cube_tester import FrankaPlaceCubeTester
from manipgen.utils.policy_testers.place_tester import FrankaPlaceTester
from manipgen.utils.policy_testers.grasp_handle_tester import FrankaGraspHandleTester
from manipgen.utils.policy_testers.open_tester import FrankaOpenTester

testers = {
    "pick_cube": FrankaPickCubeTester,
    "place_cube": FrankaPlaceCubeTester,
    "pick": FrankaPickCubeTester,
    "place": FrankaPlaceTester,
    "grasp_handle": FrankaGraspHandleTester,
    "open": FrankaOpenTester,
    "close": FrankaOpenTester,
    "reach": FrankaReachTester,
}
