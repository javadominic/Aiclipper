import unittest
from studio import render, jobs, creators

class TestSplitLayout(unittest.TestCase):
    def test_layouts_registry(self):
        self.assertIn("split", render.LAYOUTS)
        self.assertIn("Gaming split", render.LAYOUTS["split"])

    def test_cam_corners_defined(self):
        corners = ["top-left", "top-right", "bottom-left", "bottom-right"]
        for c in corners:
            self.assertIn(c, render.CAM_CORNERS)
            coords = render.CAM_CORNERS[c]
            self.assertIn("x", coords)
            self.assertIn("y", coords)
            self.assertIn("w", coords)
            self.assertIn("h", coords)
            self.assertGreater(coords["w"], 0)
            self.assertGreater(coords["h"], 0)

    def test_geometry_split(self):
        layout, h, hook_y, cap_y = render.geometry(1920, 1080, "split")
        self.assertEqual(layout, "split")
        self.assertEqual(h, render.H)
        self.assertEqual(hook_y, 240)
        self.assertEqual(cap_y, 1500)

    def test_split_filter_string(self):
        fc = render._split_filter(1920, 1080, "top-left")
        self.assertIn("crop=", fc)
        self.assertIn("vstack=inputs=3", fc)
        self.assertIn("[face][bar][game]", fc)
        self.assertIn("[v0]", fc)

    def test_clean_options_cam_corner(self):
        opts = jobs.clean_options({"layout": "split", "cam_corner": "bottom-right"})
        self.assertEqual(opts["layout"], "split")
        self.assertEqual(opts["cam_corner"], "bottom-right")

        bad_opts = jobs.clean_options({"layout": "split", "cam_corner": "invalid-corner"})
        self.assertEqual(bad_opts["cam_corner"], "top-left")

    def test_creator_defaults(self):
        self.assertIn("cam_corner", creators.DEFAULTS)
        self.assertEqual(creators.DEFAULTS["cam_corner"], "top-left")

    def test_build_ass_brand_label(self):
        ass = render.build_ass([], 0.0, 10.0, "EPIC MOMENT", "bounce", [], 200, 1450, brand="CarryisLive")
        self.assertIn("Style: Brand", ass)
        self.assertIn("CARRYISLIVE", ass)
        self.assertIn("Dialogue: 5,", ass)

if __name__ == "__main__":
    unittest.main()
