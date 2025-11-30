# 自动生成的表关系配置
# 生成时间: 2025-11-30T10:20:04.824318

RELATIONSHIPS = [
    {
        "from_table": "sgp",
        "to_table": "kgp",
        "description": "sgp left join kgp",
        "field_relation": "id references spec_goods_id",
        # 来源: TpGoodsMapper.xml - detail
    },
    {
        "from_table": "sgp",
        "to_table": "g",
        "description": "sgp left join g",
        "field_relation": "goods_id references goods_id",
        # 来源: TpGoodsMapper.xml - detail
    },
    {
        "from_table": "rm",
        "to_table": "hm",
        "description": "rm references hm",
        "field_relation": "room_id references id",
        # 来源: PsCommunityRoominfoMapper.xml - queryRoomNum
    },
    {
        "from_table": "ru",
        "to_table": "hm",
        "description": "ru references hm",
        "field_relation": "room_id references id",
        # 来源: PsCommunityRoominfoMapper.xml - queryRoomInfo
    },
    {
        "from_table": "department_community",
        "to_table": "ps_community",
        "description": "department_community left join ps_community",
        "field_relation": "xq_orgcode references event_community_no",
        # 来源: PsCommunityMapper.xml - queryCommunity
    },
    {
        "from_table": "ps_community_building",
        "to_table": "ps_community",
        "description": "ps_community_building references ps_community",
        "field_relation": "community_id references id",
        # 来源: PsCommunityMapper.xml - queryCommunity
    },
    {
        "from_table": "door_devices",
        "to_table": "ps_community",
        "description": "door_devices references ps_community",
        "field_relation": "community_id references id",
        # 来源: PsCommunityMapper.xml - queryCommunity
    },
    {
        "from_table": "parking_devices",
        "to_table": "ps_community",
        "description": "parking_devices references ps_community",
        "field_relation": "community_id references id",
        # 来源: PsCommunityMapper.xml - queryCommunity
    },
    {
        "from_table": "ps_community",
        "to_table": "ps_community_building",
        "description": "ps_community left join ps_community_building",
        "field_relation": "id references community_id",
        # 来源: PsCommunityMapper.xml - countBuildingInfo
    },
    {
        "from_table": "ps_community",
        "to_table": "ps_community_units",
        "description": "ps_community left join ps_community_units",
        "field_relation": "id references community_id",
        # 来源: PsCommunityMapper.xml - countUnitInfo
    },
    {
        "from_table": "ps_community_building",
        "to_table": "ps_community_units",
        "description": "ps_community_building left join ps_community_units",
        "field_relation": "id references building_id",
        # 来源: PsCommunityMapper.xml - queryUnitInfo
    },
    {
        "from_table": "ps_community",
        "to_table": "ps_community_roominfo",
        "description": "ps_community left join ps_community_roominfo",
        "field_relation": "id references community_id",
        # 来源: PsCommunityMapper.xml - countHouseInfo
    },
    {
        "from_table": "ps_community_roominfo",
        "to_table": "ps_community_units",
        "description": "ps_community_roominfo left join ps_community_units",
        "field_relation": "unit_id references id",
        # 来源: PsCommunityMapper.xml - queryHouseInfo
    },
    {
        "from_table": "ps_community",
        "to_table": "ps_room_user",
        "description": "ps_community left join ps_room_user",
        "field_relation": "id references community_id",
        # 来源: PsCommunityMapper.xml - countPersonInfo
    },
    {
        "from_table": "ps_community_roominfo",
        "to_table": "ps_room_user",
        "description": "ps_community_roominfo left join ps_room_user",
        "field_relation": "id references room_id",
        # 来源: PsCommunityMapper.xml - queryPersonInfo
    },
    {
        "from_table": "pcy",
        "to_table": "parking_cars",
        "description": "pcy left join parking_cars",
        "field_relation": "id references community_id",
        # 来源: PsCommunityMapper.xml - countCarInfo
    },
    {
        "from_table": "ps_room_user",
        "to_table": "ps_community",
        "description": "ps_room_user left join ps_community",
        "field_relation": "community_id references id",
        # 来源: PsRoomUserMapper.xml - queryRoomUserInfo
    },
    {
        "from_table": "ps_room_user",
        "to_table": "ps_community_roominfo",
        "description": "ps_room_user left join ps_community_roominfo",
        "field_relation": "room_id references id",
        # 来源: PsRoomUserMapper.xml - queryRoomUserInfo
    },
    {
        "from_table": "ps_community_units",
        "to_table": "ps_community_building",
        "description": "ps_community_units left join ps_community_building",
        "field_relation": "building_id references id",
        # 来源: PsRoomUserMapper.xml - queryHousingInfo
    },
    {
        "from_table": "ppc",
        "to_table": "ps_community",
        "description": "ppc references ps_community",
        "field_relation": "id references pro_company_id",
        # 来源: PsCommunityMapper.xml - queryCommunityInfo
    },
    {
        "from_table": "ps_community",
        "to_table": "door_devices",
        "description": "ps_community right join door_devices",
        "field_relation": "id references community_id",
        # 来源: PsCommunityMapper.xml - queryDoorInfo
    },
    {
        "from_table": "ps_room_user",
        "to_table": "door_record",
        "description": "ps_room_user left join door_record",
        "field_relation": "mobile references user_phone",
        # 来源: PsCommunityMapper.xml - accessRecordInfo
    },
    {
        "from_table": "ps_community",
        "to_table": "door_record",
        "description": "ps_community left join door_record",
        "field_relation": "id references community_id",
        # 来源: PsCommunityMapper.xml - accessRecordInfo
    },
    {
        "from_table": "door_devices",
        "to_table": "door_record",
        "description": "door_devices left join door_record",
        "field_relation": "device_id references device_no",
        # 来源: PsCommunityMapper.xml - accessRecordInfo
    },
    {
        "from_table": "ps_community",
        "to_table": "parking_devices",
        "description": "ps_community left join parking_devices",
        "field_relation": "id references community_id",
        # 来源: PsCommunityMapper.xml - queryCommunityPassageway
    },
    {
        "from_table": "ps_community",
        "to_table": "drr",
        "description": "ps_community left join drr",
        "field_relation": "id references community_id",
        # 来源: PsCommunityMapper.xml - queryFaceSnap
    },
    {
        "from_table": "door_devices",
        "to_table": "drr",
        "description": "door_devices left join drr",
        "field_relation": "device_id references device_no",
        # 来源: PsCommunityMapper.xml - queryFaceSnap
    },
    {
        "from_table": "parking_cars",
        "to_table": "pca",
        "description": "parking_cars left join pca",
        "field_relation": "id references community_id",
        # 来源: PsCommunityMapper.xml - queryCarInfo
    },
    {
        "from_table": "parking_user_carport",
        "to_table": "pca",
        "description": "parking_user_carport left join pca",
        "field_relation": "car_id references id",
        # 来源: PsCommunityMapper.xml - queryCarInfo
    },
    {
        "from_table": "ps_room_user",
        "to_table": "parking_user_carport",
        "description": "ps_room_user inner join parking_user_carport",
        "field_relation": "member_id references member_id",
        # 来源: PsCommunityMapper.xml - queryCarInfo
    },
    {
        "from_table": "parking_devices",
        "to_table": "parking_devices",
        "description": "parking_devices inner join parking_devices",
        "field_relation": "id references gate_id",
        # 来源: PsCommunityMapper.xml - queryCarPassageRecord
    },
    {
        "from_table": "ps_room_user",
        "to_table": "user_info",
        "description": "ps_room_user left join user_info",
        "field_relation": "mobile references mobile_number",
        # 来源: PsCommunityMapper.xml - queryPropertyManagerInfo
    },
    {
        "from_table": "door_devices",
        "to_table": "ps_capture_photos",
        "description": "door_devices left join ps_capture_photos",
        "field_relation": "device_id references device_no",
        # 来源: PsCommunityMapper.xml - queryFaceSnap
    },
    {
        "from_table": "pc",
        "to_table": "ps_capture_photos",
        "description": "pc references ps_capture_photos",
        "field_relation": "id references community_id",
        # 来源: PsCommunityMapper.xml - queryFaceSnap
    },
    {
        "from_table": "ps_property_company",
        "to_table": "ps_community",
        "description": "ps_property_company left join ps_community",
        "field_relation": "id references pro_company_id",
        # 来源: PsCommunityMapper.xml - queryCommunityInfo
    },
    {
        "from_table": "door_device_unit",
        "to_table": "door_devices",
        "description": "door_device_unit left join door_devices",
        "field_relation": "devices_id references id",
        # 来源: PsCommunityMapper.xml - queryDeviceInfo
    },
    {
        "from_table": "ps_community_units",
        "to_table": "door_device_unit",
        "description": "ps_community_units left join door_device_unit",
        "field_relation": "id references unit_id",
        # 来源: PsCommunityMapper.xml - queryDeviceInfo
    },
    {
        "from_table": "ps_community_roominfo",
        "to_table": "door_record",
        "description": "ps_community_roominfo left join door_record",
        "field_relation": "id references room_id",
        # 来源: PsCommunityMapper.xml - queryDoorEventInfo
    },
    {
        "from_table": "ps_room_user",
        "to_table": "door_record",
        "description": "ps_room_user left join door_record",
        "field_relation": "member_id references member_id",
        # 来源: PsCommunityMapper.xml - queryDoorEventInfo
    },
    {
        "from_table": "ps_community",
        "to_table": "ps_room_vistors",
        "description": "ps_community left join ps_room_vistors",
        "field_relation": "id references community_id",
        # 来源: PsCommunityMapper.xml - queryVisitorEventInfo
    },
    {
        "from_table": "ps_room_user",
        "to_table": "ps_room_vistors",
        "description": "ps_room_user left join ps_room_vistors",
        "field_relation": "member_id references member_id",
        # 来源: PsCommunityMapper.xml - queryVisitorEventInfo
    },
]
