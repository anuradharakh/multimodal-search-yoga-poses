from src.indexing.text_index import pose_descriptions


def test_one_description_per_pose_naming_both_languages():
    df = pose_descriptions()
    assert len(df) == 107 and df.pose.is_unique
    row = df.set_index("pose").loc["bakasana"]
    assert "Bakasana" in row.description and "Crow Pose" in row.description
