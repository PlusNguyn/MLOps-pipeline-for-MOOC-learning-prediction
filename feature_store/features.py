from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource
from datetime import timedelta

# Define entity
user_entity = Entity(name="userid_DI", value_type=ValueType.STRING, description="User ID")

# Define feature view
mooc_features = FeatureView(
    name="mooc_user_features",
    entities=["userid_DI"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="viewed", dtype=ValueType.INT64),
        Feature(name="explored", dtype=ValueType.INT64),
        Feature(name="grade", dtype=ValueType.FLOAT),
        Feature(name="nevents", dtype=ValueType.INT64),
        Feature(name="ndays_act", dtype=ValueType.INT64),
        Feature(name="nplay_video", dtype=ValueType.INT64),
        Feature(name="nchapters", dtype=ValueType.INT64),
        Feature(name="nforum_posts", dtype=ValueType.INT64),
        Feature(name="age", dtype=ValueType.INT64),
        Feature(name="duration_days", dtype=ValueType.INT64),
        Feature(name="activity_per_day", dtype=ValueType.FLOAT),
        Feature(name="video_per_chapter", dtype=ValueType.FLOAT),
    ],
    batch_source=FileSource(
        path="data/processed/features.csv",
        event_timestamp_column="event_timestamp",  # Need to add this
    ),
)