from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "sim" / "assets" / "genkiarm_calibrated.urdf"


def test_calibrated_urdf_is_valid_and_meshes_resolve():
    robot = ET.parse(URDF).getroot()
    assert robot.tag == "robot"
    joints = {joint.attrib["name"]: joint.attrib["type"] for joint in robot.findall("joint")}
    assert [joints[f"j{i}"] for i in range(1, 6)] == ["revolute"] * 5
    assert joints["tool_fixed"] == "fixed"
    assert joints["tcp_fixed"] == "fixed"
    for mesh in robot.findall(".//mesh"):
        assert (URDF.parent / mesh.attrib["filename"]).is_file()


def test_calibrated_urdf_has_collision_and_inertial_for_physical_links():
    robot = ET.parse(URDF).getroot()
    for link in robot.findall("link"):
        if link.attrib["name"] == "tcp":
            continue
        assert link.find("collision") is not None
        assert link.find("inertial") is not None
