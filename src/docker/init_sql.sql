CREATE DATABASE IF NOT EXISTS chat_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
use chat_db;

-- chat_db.t_alarm_info definition
DROP TABLE IF EXISTS t_alarm_info;
CREATE TABLE `t_alarm_info` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `Incident_addr` varchar(1000) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '案发地点',
  `division_name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '所属分局',
  `call_in_type` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '来电类别',
  `caller_name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '报警人姓名',
  `caller_sex` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '性别',
  `caller_age` int DEFAULT NULL COMMENT '性别',
  `caller_education` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '文化程度',
  `caller_job` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '受害人职业',
  `caller_phone_type` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '手机品牌',
  `fraud_money` float DEFAULT NULL COMMENT '涉案资金',
  `is_fraud` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '是否电诈(是，否）',
  `fraud_general_class` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '诈骗大类',
  `drainage_type` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '引流方式',
  `drainage_addr_account` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '引流地址、账号',
  `drainage_time` datetime DEFAULT NULL COMMENT '引流联系时间',
  `fraud_publicity` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '是否看（听）过反诈宣传(是，否）',
  `registration_time` datetime DEFAULT NULL COMMENT '登记时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='诈骗数据';

INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(1, '上海市浦东新区张江路123号', '浦东分局', '电话报警', '李华', '男', 28, '本科', '程序员', '华为', 5000.0, '是', '网络诈骗', '社交媒体', 'weixin:123456789', '2024-01-01 10:00:00', '是', '2024-01-05 10:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(2, '上海市浦东新区世纪公园', '浦东分局', '现场报警', '王芳', '女', 32, '硕士', '设计师', '苹果', 10000.0, '是', '冒充公检法', '电话', '+8613800138000', '2024-01-02 11:00:00', '否', '2024-01-10 11:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(3, '北京市朝阳区三里屯太古里', '朝阳分局', '现场报警', '刘丽', '女', 29, '本科', '教师', '三星', 8000.0, '否', '其他', '邮件', 'example@example.com', '2024-01-03 15:00:00', '否', '2024-01-15 15:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(4, '广州市天河区天河北路', '天河分局', '电话报警', '张强', '男', 30, '本科', '程序员', '华为', 7000.0, '是', '网络诈骗', '社交媒体', 'weixin:123456789', '2024-01-04 10:00:00', '是', '2024-01-20 10:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(5, '深圳市南山区科技园', '南山分局', '电话报警', '王磊', '男', 35, '高中', '司机', '小米', 4000.0, '否', '其他', '短信', '13700001111', '2024-01-05 12:00:00', '是', '2024-01-25 12:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(6, '上海市浦东新区陆家嘴环路', '浦东分局', '电话报警', '陈晓', '女', 25, '本科', '销售', 'OPPO', 2000.0, '是', '购物退款', '网站链接', 'http://example.com', '2024-02-01 13:00:00', '否', '2024-02-05 13:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(7, '上海市浦东新区花木路', '浦东分局', '电话报警', '周杰', '男', 30, '大专', '工程师', 'VIVO', 15000.0, '是', '投资理财', '电话', '+8613700002222', '2024-02-02 14:00:00', '是', '2024-02-10 14:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(8, '北京市朝阳区国贸CBD', '朝阳分局', '电话报警', '张伟', '男', 35, '硕士', '产品经理', '华为', 12000.0, '是', '冒充亲友', '电话', '+8613700003333', '2024-02-03 16:00:00', '是', '2024-02-15 16:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(9, '广州市天河区珠江新城', '天河分局', '现场报警', '李娜', '女', 28, '硕士', '设计师', '苹果', 12000.0, '是', '冒充公检法', '电话', '+8613800138000', '2024-02-04 11:00:00', '否', '2024-02-20 11:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(10, '上海市浦东新区周康路', '浦东分局', '电话报警', '赵雷', '男', 45, '高中', '司机', '小米', 3000.0, '否', '其他', '短信', '13700001111', '2024-03-01 12:00:00', '是', '2024-03-05 12:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(11, '北京市朝阳区工体北路', '朝阳分局', '电话报警', '孙娜', '女', 27, '本科', '财务', '苹果', 5000.0, '否', '其他', '社交媒体', 'weixin:987654321', '2024-03-02 17:00:00', '否', '2024-03-10 17:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(12, '广州市天河区体育西路', '天河分局', '电话报警', '王磊', '男', 35, '高中', '司机', '小米', 4000.0, '否', '其他', '短信', '13700001111', '2024-03-03 12:00:00', '是', '2024-03-15 12:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(13, '深圳市南山区南头关', '南山分局', '电话报警', '赵晓', '女', 26, '本科', '销售', 'OPPO', 3000.0, '是', '购物退款', '网站链接', 'http://example.com', '2024-04-01 13:00:00', '否', '2024-04-05 13:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(14, '成都市武侯区武侯祠大街', '武侯分局', '电话报警', '周涛', '男', 31, '大专', '工程师', 'VIVO', 16000.0, '是', '投资理财', '电话', '+8613700002222', '2024-04-02 14:00:00', '是', '2024-04-10 14:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(15, '成都市锦江区春熙路', '锦江分局', '电话报警', '刘洋', '女', 30, '本科', '教师', '三星', 9000.0, '否', '其他', '邮件', 'example@example.com', '2024-05-01 15:00:00', '否', '2024-05-05 15:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(16, '上海市徐汇区漕溪路', '徐汇分局', '电话报警', '张明', '男', 36, '硕士', '产品经理', '华为', 13000.0, '是', '冒充亲友', '电话', '+8613700003333', '2024-06-01 16:00:00', '是', '2024-06-05 16:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(17, '上海市徐汇区徐家汇路', '徐汇分局', '电话报警', '孙莉', '女', 28, '本科', '财务', '苹果', 6000.0, '否', '其他', '社交媒体', 'weixin:987654321', '2024-06-02 17:00:00', '否', '2024-06-10 17:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(18, '上海市徐汇区龙阳路', '徐汇分局', '电话报警', '黄梅', '女', 38, '本科', '经理', '华为', 21000.0, '是', '冒充公检法', '电话', '+8613700004444', '2024-06-03 18:00:00', '是', '2024-06-15 18:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(19, '上海市徐汇区漕宝路', '徐汇分局', '电话报警', '李刚', '男', 34, '本科', '行政', '小米', 7000.0, '否', '其他', '短信', '13700005555', '2024-06-04 19:00:00', '是', '2024-06-20 19:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(20, '上海市徐汇区漕河泾开发区', '徐汇分局', '电话报警', '张强', '男', 30, '本科', '程序员', '华为', 7000.0, '是', '网络诈骗', '社交媒体', 'weixin:123456789', '2024-06-05 10:00:00', '是', '2024-06-25 10:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(21, '北京市海淀区中关村大街', '海淀分局', '现场报警', '李娜', '女', 28, '硕士', '设计师', '苹果', 12000.0, '是', '冒充公检法', '电话', '+8613800138000', '2024-07-01 11:00:00', '否', '2024-07-05 11:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(22, '北京市海淀区五道口', '海淀分局', '电话报警', '王磊', '男', 35, '高中', '司机', '小米', 4000.0, '否', '其他', '短信', '13700001111', '2024-07-02 12:00:00', '是', '2024-07-10 12:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(23, '北京市海淀区西直门', '海淀分局', '电话报警', '赵晓', '女', 26, '本科', '销售', 'OPPO', 3000.0, '是', '购物退款', '网站链接', 'http://example.com', '2024-07-03 13:00:00', '否', '2024-07-15 13:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(24, '北京市海淀区学院路', '海淀分局', '电话报警', '周涛', '男', 31, '大专', '工程师', 'VIVO', 16000.0, '是', '投资理财', '电话', '+8613700002222', '2024-07-04 14:00:00', '是', '2024-07-20 14:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(25, '广州市白云区白云大道', '白云分局', '电话报警', '刘洋', '女', 30, '本科', '教师', '三星', 9000.0, '否', '其他', '邮件', 'example@example.com', '2024-08-01 15:00:00', '否', '2024-08-05 15:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(26, '广州市白云区金沙洲', '白云分局', '电话报警', '张明', '男', 36, '硕士', '产品经理', '华为', 13000.0, '是', '冒充亲友', '电话', '+8613700003333', '2024-08-02 16:00:00', '是', '2024-08-10 16:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(27, '广州市白云区同德围', '白云分局', '电话报警', '孙莉', '女', 28, '本科', '财务', '苹果', 6000.0, '否', '其他', '社交媒体', 'weixin:987654321', '2024-08-03 17:00:00', '否', '2024-08-15 17:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(28, '深圳市福田区福华路', '福田分局', '电话报警', '黄梅', '女', 38, '本科', '经理', '华为', 21000.0, '是', '冒充公检法', '电话', '+8613700004444', '2024-09-01 18:00:00', '是', '2024-09-05 18:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(29, '深圳市福田区莲花路', '福田分局', '电话报警', '李刚', '男', 34, '本科', '行政', '小米', 7000.0, '否', '其他', '短信', '13700005555', '2024-09-02 19:00:00', '是', '2024-09-10 19:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(30, '成都市锦江区春熙路', '锦江分局', '电话报警', '刘洋', '女', 30, '本科', '教师', '三星', 9000.0, '否', '其他', '邮件', 'example@example.com', '2024-10-01 15:00:00', '否', '2024-10-05 15:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(31, '上海市浦东新区张江路123号', '浦东分局', '电话报警', '李华', '男', 28, '本科', '程序员', '华为', 5000.0, '是', '网络诈骗', '社交媒体', 'weixin:123456789', '2024-11-01 10:00:00', '是', '2024-11-05 10:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(32, '上海市浦东新区世纪公园', '浦东分局', '现场报警', '王芳', '女', 32, '硕士', '设计师', '苹果', 10000.0, '是', '冒充公检法', '电话', '+8613800138000', '2024-11-02 11:00:00', '否', '2024-11-10 11:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(33, '北京市朝阳区三里屯太古里', '朝阳分局', '现场报警', '刘丽', '女', 29, '本科', '教师', '三星', 8000.0, '否', '其他', '邮件', 'example@example.com', '2024-11-03 15:00:00', '否', '2024-11-15 15:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(34, '广州市天河区天河北路', '天河分局', '电话报警', '张强', '男', 30, '本科', '程序员', '华为', 7000.0, '是', '网络诈骗', '社交媒体', 'weixin:123456789', '2024-11-04 10:00:00', '是', '2024-11-20 10:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(35, '深圳市南山区科技园', '南山分局', '电话报警', '王磊', '男', 35, '高中', '司机', '小米', 4000.0, '否', '其他', '短信', '13700001111', '2024-11-05 12:00:00', '是', '2024-11-25 12:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(36, '上海市徐汇区漕溪路', '徐汇分局', '电话报警', '张明', '男', 36, '硕士', '产品经理', '华为', 13000.0, '是', '冒充亲友', '电话', '+8613700003333', '2024-12-01 16:00:00', '是', '2024-12-05 16:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(37, '上海市徐汇区徐家汇路', '徐汇分局', '电话报警', '孙莉', '女', 28, '本科', '财务', '苹果', 6000.0, '否', '其他', '社交媒体', 'weixin:987654321', '2024-12-02 17:00:00', '否', '2024-12-10 17:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(38, '上海市徐汇区龙阳路', '徐汇分局', '电话报警', '黄梅', '女', 38, '本科', '经理', '华为', 21000.0, '是', '冒充公检法', '电话', '+8613700004444', '2024-12-03 18:00:00', '是', '2024-12-15 18:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(39, '上海市徐汇区漕宝路', '徐汇分局', '电话报警', '李刚', '男', 34, '本科', '行政', '小米', 7000.0, '否', '其他', '短信', '13700005555', '2024-12-04 19:00:00', '是', '2024-12-20 19:30:00');
INSERT INTO chat_db.t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(40, '上海市徐汇区漕河泾开发区', '徐汇分局', '电话报警', '张强', '男', 30, '本科', '程序员', '华为', 7000.0, '是', '网络诈骗', '社交媒体', 'weixin:123456789', '2024-12-05 10:00:00', '是', '2024-12-25 10:30:00');

-- chat_db.t_user definition
DROP TABLE IF EXISTS t_user;
CREATE TABLE `t_user` (
  `id` int NOT NULL AUTO_INCREMENT,
  `userName` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '用户名称',
  `password` varchar(300) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '密码',
  `mobile` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '手机号',
  `createTime` datetime DEFAULT NULL COMMENT '创建时间',
  `updateTime` datetime DEFAULT NULL COMMENT '修改时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

INSERT INTO chat_db.t_user
(id, userName, password, mobile, createTime, updateTime)
VALUES(1, 'admin', '123456', NULL, '2024-01-15 15:30:00', '2024-01-15 15:30:00');

-- chat_db.t_user_qa_record definition
DROP TABLE IF EXISTS t_user_qa_record;
CREATE TABLE `t_user_qa_record` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `user_id` int DEFAULT NULL COMMENT '用户id',
  `uuid` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL COMMENT '自定义id',
  `conversation_id` varchar(100) DEFAULT NULL COMMENT 'diy/对话id',
  `message_id` varchar(100) DEFAULT NULL COMMENT 'dify/消息id',
  `task_id` varchar(100) DEFAULT NULL COMMENT 'dify/任务id',
  `chat_id` varchar(100) DEFAULT NULL COMMENT '对话id',
  `question` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '用户问题',
  `to2_answer` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '大模型答案',
  `to4_answer` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '业务数据',
  `qa_type` varchar(100) DEFAULT NULL COMMENT '问答类型',
  `file_key` varchar(100) DEFAULT NULL COMMENT '文件minio/key',
  `create_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=567 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='问答记录表';

DROP TABLE IF EXISTS  t_demand_doc_meta;
CREATE TABLE `t_demand_doc_meta` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int DEFAULT NULL COMMENT '用户id',
  `demand_id` int DEFAULT NULL COMMENT '项目id',
  `page_title` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '文档段落标题名称',
  `page_content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '文档段落内容',
  `create_time` timestamp NULL DEFAULT NULL,
  `update_time` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=23 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='需求文档元信息';

-- chat_db.t_demand_manager definition
DROP TABLE IF EXISTS  t_demand_manager;
CREATE TABLE `t_demand_manager` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int DEFAULT NULL COMMENT '用户id',
  `doc_name` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '项目名称',
  `doc_desc` varchar(200) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '项目简介',
  `file_key` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '文件minio key',
  `fun_num` int DEFAULT '0' COMMENT '功能数',
  `create_time` timestamp NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` timestamp NULL DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=35 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='测试助手-需求文档管理';

-- chat_db.t_report_info definition
drop table if exists t_report_info;
CREATE TABLE `t_report_info` (
  `id` int NOT NULL AUTO_INCREMENT,
  `title` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '报告名称',
  `markdown` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '报告内容',
  `create_time` datetime DEFAULT NULL COMMENT '创建时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='报告记录表';

INSERT INTO chat_db.t_report_info
(id, title, markdown, create_time)
VALUES(1, '2024年大模型排行报告', '### 2024年大模型排行报告

#### 概述

2024年，随着人工智能技术的迅速发展，大模型在自然语言处理、图像识别等多个领域取得了显著进展。这些模型不仅在学术界产生了深远影响，也在工业界得到了广泛应用。本报告旨在综合评估2024年的主要大模型，从技术性能、应用场景、用户评价等多个维度进行分析，以帮助读者更好地了解各模型的特点和适用场景。


#### 排行榜

以下是根据上述评估维度整理出的2024年大模型排行榜单及性能参数表：

| 排名 | 模型名称         | 参数量 (B) | 训练数据集规模 (TB) | 推理速度 (tokens/s) | 上下文窗口长度 (tokens) | 主要应用场景           | 用户评价 | 价格与可用性 |
|------|------------------|------------|---------------------|---------------------|-------------------------|------------------------|----------|--------------|
| 1    | 百川智能-百川三  | 175        | 400                 | 1000                | 8192                    | 文本生成、代码辅助     | 高       | 中等         |
| 2    | 谷歌-PaLM 2      | 1100       | 500                 | 800                 | 16384                   | 多语言翻译、知识问答   | 非常高   | 高           |
| 3    | 微软-Turing NLG  | 1000       | 350                 | 700                 | 4096                    | 企业级应用             | 高       | 较高         |
| 4    | Meta-LLaMA 2     | 1300       | 450                 | 900                 | 8192                    | 社交媒体内容创作       | 高       | 中等         |
| 5    | 阿里云-Qwen      | 260        | 300                 | 1200                | 2048                    | 客服机器人             | 高       | 低           |
| 6    | 百度-文心一言    | 150        | 250                 | 950                 | 4096                    | 自然语言生成、智能问答 | 高       | 低           |
| 7    | 字节跳动-豆包    | 120        | 200                 | 850                 | 2048                    | 语音交互               | 高       | 低           |
| 8    | 昆仑万维-天工AI  | 100        | 150                 | 750                 | 4096                    | 图像识别、自然语言处理 | 中       | 低           |
| 9    | 北京智谱华章-智谱清言 | 80        | 100                 | 600                 | 3072                    | 自然语言处理           | 中       | 低           |
| 10   | 科大讯飞-讯飞星火 | 90        | 120                 | 800                 | 2048                    | 语音识别、语音合成     | 高       | 低           |

#### 详细分析

1. **技术性能**
   - **参数量**：参数量越大通常意味着更强的表达能力和泛化能力。例如，谷歌的PaLM 2拥有1100亿参数，是目前参数量最大的模型之一，具有强大的多语言翻译和知识问答能力。
   - **训练数据集规模**：更大的训练数据集有助于模型学习更多的知识和模式。例如，百川智能的百川三使用了400TB的数据集，使其在文本生成和代码辅助方面表现出色。
   - **推理速度**：推理速度是衡量模型在实际应用中性能的重要指标。例如，阿里云的Qwen在推理速度上表现出色，达到了1200 tokens/s。
   - **上下文窗口长度**：上下文窗口长度决定了模型能够处理的最长文本长度。例如，PaLM 2的上下文窗口长度为16384 tokens，适合处理长文档和复杂任务。

2. **应用范围**
   - **自然语言生成**：许多大模型在自然语言生成方面表现出色，如百川智能的百川三和百度的文心一言。
   - **翻译**：多语言翻译是大模型的重要应用场景之一，如谷歌的PaLM 2和微软的Turing NLG。
   - **对话系统**：大模型在对话系统中的应用也越来越广泛，如阿里云的Qwen和科大讯飞的讯飞星火。
   - **图像生成**：部分模型在图像生成方面也有出色表现，如昆仑万维的天工AI。
   - **多模态融合**：多模态融合是未来大模型发展的趋势之一，如Meta的LLaMA 2。

3. **用户评价**
   - **用户反馈**：用户反馈是评估模型实际效果的重要依据。从表中可以看出，所有上榜模型都获得了较高的用户评价，尤其是谷歌的PaLM 2和阿里云的Qwen。
   - **专业评测**：专业评测机构的报告也提供了重要参考。例如，根据《中文大模型基准测评2024年度4月报告》，百川智能的百川三在多个维度上表现优秀，排名第一。

4. **价格与可用性**
   - **成本效益**：对于预算有限或对成本敏感的用户来说，选择性价比高的模型尤为重要。例如，阿里云的Qwen不仅价格亲民，而且易于集成到现有系统中。
   - **市场可获得性**：部分模型可能需要特定的硬件支持或授权许可，影响其市场可获得性。例如，谷歌的PaLM 2虽然性能强大，但获取成本较高。

#### 结论

综合考虑技术性能、应用场景、用户评价和价格与可用性，2024年的大模型市场呈现出多样化的格局。不同模型在各自领域内具有独特的优势，用户可以根据具体需求选择合适的模型。希望本报告能为您的决策提供有价值的参考。

---

参考资料：
-  2024年4月《中文大模型基准测评》报告，详细评估了多个大模型的技术性能和应用场景。', '2024-11-14 00:00:00');


-- chat_db.view_alarm_detail source

CREATE OR REPLACE
ALGORITHM = UNDEFINED VIEW `chat_db`.`view_alarm_detail` AS
select
    `chat_db`.`t_alarm_info`.`Incident_addr` AS `案发地点`,
    `chat_db`.`t_alarm_info`.`division_name` AS `所属分局`,
    `chat_db`.`t_alarm_info`.`call_in_type` AS `来电类别`,
    `chat_db`.`t_alarm_info`.`caller_name` AS `报警人姓名`,
    `chat_db`.`t_alarm_info`.`caller_sex` AS `性别`,
    `chat_db`.`t_alarm_info`.`caller_age` AS `年龄`,
    `chat_db`.`t_alarm_info`.`caller_education` AS `文化程度`,
    `chat_db`.`t_alarm_info`.`caller_job` AS `受害人职业`,
    `chat_db`.`t_alarm_info`.`caller_phone_type` AS `手机品牌`,
    `chat_db`.`t_alarm_info`.`fraud_money` AS `涉案资金`,
    `chat_db`.`t_alarm_info`.`is_fraud` AS `是否电诈`,
    `chat_db`.`t_alarm_info`.`fraud_general_class` AS `诈骗类型`,
    `chat_db`.`t_alarm_info`.`drainage_type` AS `引流方式`,
    `chat_db`.`t_alarm_info`.`drainage_addr_account` AS `引流地址`,
    `chat_db`.`t_alarm_info`.`drainage_time` AS `引流联系时间`,
    `chat_db`.`t_alarm_info`.`fraud_publicity` AS `是否看过反诈宣传`,
    `chat_db`.`t_alarm_info`.`registration_time` AS `登记时间`
from
    `chat_db`.`t_alarm_info`;









