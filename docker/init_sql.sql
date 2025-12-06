CREATE DATABASE IF NOT EXISTS chat_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
use chat_db;

-- t_alarm_info definition
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

INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(1, '上海市浦东新区张江路123号', '浦东分局', '电话报警', '李华', '男', 28, '本科', '程序员', '华为', 5000.0, '是', '网络诈骗', '社交媒体', 'weixin:123456789', '2024-01-01 10:00:00', '是', '2024-01-05 10:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(2, '上海市浦东新区世纪公园', '浦东分局', '现场报警', '王芳', '女', 32, '硕士', '设计师', '苹果', 10000.0, '是', '冒充公检法', '电话', '+8613800138000', '2024-01-02 11:00:00', '否', '2024-01-10 11:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(3, '北京市朝阳区三里屯太古里', '朝阳分局', '现场报警', '刘丽', '女', 29, '本科', '教师', '三星', 8000.0, '否', '其他', '邮件', 'example@example.com', '2024-01-03 15:00:00', '否', '2024-01-15 15:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(4, '广州市天河区天河北路', '天河分局', '电话报警', '张强', '男', 30, '本科', '程序员', '华为', 7000.0, '是', '网络诈骗', '社交媒体', 'weixin:123456789', '2024-01-04 10:00:00', '是', '2024-01-20 10:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(5, '深圳市南山区科技园', '南山分局', '电话报警', '王磊', '男', 35, '高中', '司机', '小米', 4000.0, '否', '其他', '短信', '13700001111', '2024-01-05 12:00:00', '是', '2024-01-25 12:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(6, '上海市浦东新区陆家嘴环路', '浦东分局', '电话报警', '陈晓', '女', 25, '本科', '销售', 'OPPO', 2000.0, '是', '购物退款', '网站链接', 'http://example.com', '2024-02-01 13:00:00', '否', '2024-02-05 13:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(7, '上海市浦东新区花木路', '浦东分局', '电话报警', '周杰', '男', 30, '大专', '工程师', 'VIVO', 15000.0, '是', '投资理财', '电话', '+8613700002222', '2024-02-02 14:00:00', '是', '2024-02-10 14:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(8, '北京市朝阳区国贸CBD', '朝阳分局', '电话报警', '张伟', '男', 35, '硕士', '产品经理', '华为', 12000.0, '是', '冒充亲友', '电话', '+8613700003333', '2024-02-03 16:00:00', '是', '2024-02-15 16:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(9, '广州市天河区珠江新城', '天河分局', '现场报警', '李娜', '女', 28, '硕士', '设计师', '苹果', 12000.0, '是', '冒充公检法', '电话', '+8613800138000', '2024-02-04 11:00:00', '否', '2024-02-20 11:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(10, '上海市浦东新区周康路', '浦东分局', '电话报警', '赵雷', '男', 45, '高中', '司机', '小米', 3000.0, '否', '其他', '短信', '13700001111', '2024-03-01 12:00:00', '是', '2024-03-05 12:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(11, '北京市朝阳区工体北路', '朝阳分局', '电话报警', '孙娜', '女', 27, '本科', '财务', '苹果', 5000.0, '否', '其他', '社交媒体', 'weixin:987654321', '2024-03-02 17:00:00', '否', '2024-03-10 17:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(12, '广州市天河区体育西路', '天河分局', '电话报警', '王磊', '男', 35, '高中', '司机', '小米', 4000.0, '否', '其他', '短信', '13700001111', '2024-03-03 12:00:00', '是', '2024-03-15 12:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(13, '深圳市南山区南头关', '南山分局', '电话报警', '赵晓', '女', 26, '本科', '销售', 'OPPO', 3000.0, '是', '购物退款', '网站链接', 'http://example.com', '2024-04-01 13:00:00', '否', '2024-04-05 13:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(14, '成都市武侯区武侯祠大街', '武侯分局', '电话报警', '周涛', '男', 31, '大专', '工程师', 'VIVO', 16000.0, '是', '投资理财', '电话', '+8613700002222', '2024-04-02 14:00:00', '是', '2024-04-10 14:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(15, '成都市锦江区春熙路', '锦江分局', '电话报警', '刘洋', '女', 30, '本科', '教师', '三星', 9000.0, '否', '其他', '邮件', 'example@example.com', '2024-05-01 15:00:00', '否', '2024-05-05 15:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(16, '上海市徐汇区漕溪路', '徐汇分局', '电话报警', '张明', '男', 36, '硕士', '产品经理', '华为', 13000.0, '是', '冒充亲友', '电话', '+8613700003333', '2024-06-01 16:00:00', '是', '2024-06-05 16:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(17, '上海市徐汇区徐家汇路', '徐汇分局', '电话报警', '孙莉', '女', 28, '本科', '财务', '苹果', 6000.0, '否', '其他', '社交媒体', 'weixin:987654321', '2024-06-02 17:00:00', '否', '2024-06-10 17:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(18, '上海市徐汇区龙阳路', '徐汇分局', '电话报警', '黄梅', '女', 38, '本科', '经理', '华为', 21000.0, '是', '冒充公检法', '电话', '+8613700004444', '2024-06-03 18:00:00', '是', '2024-06-15 18:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(19, '上海市徐汇区漕宝路', '徐汇分局', '电话报警', '李刚', '男', 34, '本科', '行政', '小米', 7000.0, '否', '其他', '短信', '13700005555', '2024-06-04 19:00:00', '是', '2024-06-20 19:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(20, '上海市徐汇区漕河泾开发区', '徐汇分局', '电话报警', '张强', '男', 30, '本科', '程序员', '华为', 7000.0, '是', '网络诈骗', '社交媒体', 'weixin:123456789', '2024-06-05 10:00:00', '是', '2024-06-25 10:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(21, '北京市海淀区中关村大街', '海淀分局', '现场报警', '李娜', '女', 28, '硕士', '设计师', '苹果', 12000.0, '是', '冒充公检法', '电话', '+8613800138000', '2024-07-01 11:00:00', '否', '2024-07-05 11:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(22, '北京市海淀区五道口', '海淀分局', '电话报警', '王磊', '男', 35, '高中', '司机', '小米', 4000.0, '否', '其他', '短信', '13700001111', '2024-07-02 12:00:00', '是', '2024-07-10 12:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(23, '北京市海淀区西直门', '海淀分局', '电话报警', '赵晓', '女', 26, '本科', '销售', 'OPPO', 3000.0, '是', '购物退款', '网站链接', 'http://example.com', '2024-07-03 13:00:00', '否', '2024-07-15 13:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(24, '北京市海淀区学院路', '海淀分局', '电话报警', '周涛', '男', 31, '大专', '工程师', 'VIVO', 16000.0, '是', '投资理财', '电话', '+8613700002222', '2024-07-04 14:00:00', '是', '2024-07-20 14:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(25, '广州市白云区白云大道', '白云分局', '电话报警', '刘洋', '女', 30, '本科', '教师', '三星', 9000.0, '否', '其他', '邮件', 'example@example.com', '2024-08-01 15:00:00', '否', '2024-08-05 15:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(26, '广州市白云区金沙洲', '白云分局', '电话报警', '张明', '男', 36, '硕士', '产品经理', '华为', 13000.0, '是', '冒充亲友', '电话', '+8613700003333', '2024-08-02 16:00:00', '是', '2024-08-10 16:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(27, '广州市白云区同德围', '白云分局', '电话报警', '孙莉', '女', 28, '本科', '财务', '苹果', 6000.0, '否', '其他', '社交媒体', 'weixin:987654321', '2024-08-03 17:00:00', '否', '2024-08-15 17:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(28, '深圳市福田区福华路', '福田分局', '电话报警', '黄梅', '女', 38, '本科', '经理', '华为', 21000.0, '是', '冒充公检法', '电话', '+8613700004444', '2024-09-01 18:00:00', '是', '2024-09-05 18:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(29, '深圳市福田区莲花路', '福田分局', '电话报警', '李刚', '男', 34, '本科', '行政', '小米', 7000.0, '否', '其他', '短信', '13700005555', '2024-09-02 19:00:00', '是', '2024-09-10 19:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(30, '成都市锦江区春熙路', '锦江分局', '电话报警', '刘洋', '女', 30, '本科', '教师', '三星', 9000.0, '否', '其他', '邮件', 'example@example.com', '2024-10-01 15:00:00', '否', '2024-10-05 15:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(31, '上海市浦东新区张江路123号', '浦东分局', '电话报警', '李华', '男', 28, '本科', '程序员', '华为', 5000.0, '是', '网络诈骗', '社交媒体', 'weixin:123456789', '2024-11-01 10:00:00', '是', '2024-11-05 10:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(32, '上海市浦东新区世纪公园', '浦东分局', '现场报警', '王芳', '女', 32, '硕士', '设计师', '苹果', 10000.0, '是', '冒充公检法', '电话', '+8613800138000', '2024-11-02 11:00:00', '否', '2024-11-10 11:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(33, '北京市朝阳区三里屯太古里', '朝阳分局', '现场报警', '刘丽', '女', 29, '本科', '教师', '三星', 8000.0, '否', '其他', '邮件', 'example@example.com', '2024-11-03 15:00:00', '否', '2024-11-15 15:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(34, '广州市天河区天河北路', '天河分局', '电话报警', '张强', '男', 30, '本科', '程序员', '华为', 7000.0, '是', '网络诈骗', '社交媒体', 'weixin:123456789', '2024-11-04 10:00:00', '是', '2024-11-20 10:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(35, '深圳市南山区科技园', '南山分局', '电话报警', '王磊', '男', 35, '高中', '司机', '小米', 4000.0, '否', '其他', '短信', '13700001111', '2024-11-05 12:00:00', '是', '2024-11-25 12:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(36, '上海市徐汇区漕溪路', '徐汇分局', '电话报警', '张明', '男', 36, '硕士', '产品经理', '华为', 13000.0, '是', '冒充亲友', '电话', '+8613700003333', '2024-12-01 16:00:00', '是', '2024-12-05 16:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(37, '上海市徐汇区徐家汇路', '徐汇分局', '电话报警', '孙莉', '女', 28, '本科', '财务', '苹果', 6000.0, '否', '其他', '社交媒体', 'weixin:987654321', '2024-12-02 17:00:00', '否', '2024-12-10 17:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(38, '上海市徐汇区龙阳路', '徐汇分局', '电话报警', '黄梅', '女', 38, '本科', '经理', '华为', 21000.0, '是', '冒充公检法', '电话', '+8613700004444', '2024-12-03 18:00:00', '是', '2024-12-15 18:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(39, '上海市徐汇区漕宝路', '徐汇分局', '电话报警', '李刚', '男', 34, '本科', '行政', '小米', 7000.0, '否', '其他', '短信', '13700005555', '2024-12-04 19:00:00', '是', '2024-12-20 19:30:00');
INSERT INTO t_alarm_info
(id, Incident_addr, division_name, call_in_type, caller_name, caller_sex, caller_age, caller_education, caller_job, caller_phone_type, fraud_money, is_fraud, fraud_general_class, drainage_type, drainage_addr_account, drainage_time, fraud_publicity, registration_time)
VALUES(40, '上海市徐汇区漕河泾开发区', '徐汇分局', '电话报警', '张强', '男', 30, '本科', '程序员', '华为', 7000.0, '是', '网络诈骗', '社交媒体', 'weixin:123456789', '2024-12-05 10:00:00', '是', '2024-12-25 10:30:00');

-- t_user definition
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

INSERT INTO t_user
(id, userName, password, mobile, createTime, updateTime)
VALUES(1, 'admin', '123456', NULL, '2024-01-15 15:30:00', '2024-01-15 15:30:00');

-- t_user_qa_record definition
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
   `file_key` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '文件minio/key',
  `create_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=567 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='问答记录表';

-- t_customers definition
DROP TABLE IF EXISTS `t_customers`;
CREATE TABLE `t_customers` (
  `customer_id` int NOT NULL AUTO_INCREMENT COMMENT '客户ID',
  `customer_name` varchar(100) NOT NULL COMMENT '客户姓名',
  `phone` varchar(20) DEFAULT NULL COMMENT '联系电话',
  `email` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL COMMENT '电子邮箱',
  `address` text COMMENT '地址',
  `city` varchar(50) DEFAULT NULL COMMENT '城市',
  `country` varchar(50) DEFAULT '中国' COMMENT '国家',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`customer_id`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='客户信息表';

INSERT INTO t_customers
(customer_id, customer_name, phone, email, address, city, country, created_at, updated_at)
VALUES(1, '张三', '13800138001', 'zhangsan@email.com', '北京市朝阳区建国路88号', '北京', '中国', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_customers
(customer_id, customer_name, phone, email, address, city, country, created_at, updated_at)
VALUES(2, '李四', '13900139002', 'lisi@email.com', '上海市浦东新区陆家嘴环路1000号', '上海', '中国', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_customers
(customer_id, customer_name, phone, email, address, city, country, created_at, updated_at)
VALUES(3, '王五', '13700137003', 'wangwu@email.com', '广州市天河区珠江新城华夏路10号', '广州', '中国', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_customers
(customer_id, customer_name, phone, email, address, city, country, created_at, updated_at)
VALUES(4, '赵六', '13600136004', 'zhaoliu@email.com', '深圳市南山区科技园北区道康路55号', '深圳', '中国', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_customers
(customer_id, customer_name, phone, email, address, city, country, created_at, updated_at)
VALUES(5, '钱七', '13500135005', 'qianqi@email.com', '杭州市西湖区文三路999号', '杭州', '中国', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_customers
(customer_id, customer_name, phone, email, address, city, country, created_at, updated_at)
VALUES(6, '孙八', '13400134006', 'sunba@email.com', '成都市高新区天府大道中段888号', '成都', '中国', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_customers
(customer_id, customer_name, phone, email, address, city, country, created_at, updated_at)
VALUES(7, '周九', '13300133007', 'zhoujiu@email.com', '西安市雁塔区科技路33号', '西安', '中国', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_customers
(customer_id, customer_name, phone, email, address, city, country, created_at, updated_at)
VALUES(8, '吴十', '13200132008', 'wushi@email.com', '南京市鼓楼区中山路123号', '南京', '中国', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_customers
(customer_id, customer_name, phone, email, address, city, country, created_at, updated_at)
VALUES(9, '郑一', '13100131009', 'zhengyi@email.com', '武汉市武昌区解放路456号', '武汉', '中国', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_customers
(customer_id, customer_name, phone, email, address, city, country, created_at, updated_at)
VALUES(10, '王二', '13000130010', 'wanger@email.com', '天津市和平区南京路789号', '天津', '中国', '2025-08-24 20:44:41', '2025-08-24 20:44:41');


-- t_order_details definition
DROP TABLE IF EXISTS `t_order_details`;
CREATE TABLE `t_order_details` (
  `detail_id` int NOT NULL AUTO_INCREMENT COMMENT '明细ID',
  `order_id` int NOT NULL COMMENT '订单ID',
  `product_id` int NOT NULL COMMENT '产品ID',
  `quantity` int NOT NULL COMMENT '销售数量',
  `unit_price` decimal(10,2) NOT NULL COMMENT '销售时的单价',
  `line_total` decimal(12,2) NOT NULL COMMENT '行小计（quantity * unit_price）',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`detail_id`),
  UNIQUE KEY `uk_order_product` (`order_id`,`product_id`),
  KEY `product_id` (`product_id`)
) ENGINE=InnoDB AUTO_INCREMENT=38 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='销售订单明细表';


INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(1, 1, 1, 1, 8999.00, 8999.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(2, 1, 13, 3, 7.80, 23.40, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(3, 2, 2, 1, 9499.00, 9499.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(4, 2, 11, 6, 3.50, 21.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(5, 3, 4, 2, 1899.00, 3798.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(6, 3, 15, 2, 3.80, 7.60, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(7, 4, 7, 1, 799.00, 799.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(8, 4, 12, 4, 3.20, 12.80, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(9, 5, 3, 1, 8499.00, 8499.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(10, 5, 14, 1, 12.50, 12.50, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(11, 6, 8, 3, 699.00, 2097.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(12, 6, 11, 10, 3.50, 35.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(13, 7, 9, 5, 199.00, 995.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(14, 7, 13, 2, 7.80, 15.60, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(15, 8, 5, 1, 7999.00, 7999.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(16, 8, 12, 8, 3.20, 25.60, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(17, 9, 6, 1, 899.00, 899.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(18, 9, 15, 5, 3.80, 19.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(19, 10, 10, 1, 1299.00, 1299.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(20, 10, 11, 12, 3.50, 42.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(21, 11, 1, 1, 8999.00, 8999.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(22, 12, 2, 1, 9499.00, 9499.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(23, 13, 4, 1, 1899.00, 1899.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(24, 13, 14, 2, 12.50, 25.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(25, 14, 7, 2, 799.00, 1598.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(26, 14, 13, 1, 7.80, 7.80, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(27, 15, 3, 1, 8499.00, 8499.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(28, 15, 12, 5, 3.20, 16.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(29, 16, 9, 2, 199.00, 398.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(30, 16, 11, 8, 3.50, 28.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(31, 17, 5, 1, 7999.00, 7999.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(32, 17, 15, 3, 3.80, 11.40, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(33, 18, 6, 1, 899.00, 899.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(34, 19, 9, 3, 199.00, 597.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(35, 19, 14, 1, 12.50, 12.50, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(36, 20, 10, 1, 1299.00, 1299.00, '2025-08-24 20:44:41');
INSERT INTO t_order_details
(detail_id, order_id, product_id, quantity, unit_price, line_total, created_at)
VALUES(37, 20, 13, 2, 7.80, 15.60, '2025-08-24 20:44:41');



-- t_products definition
DROP TABLE IF EXISTS `t_products`;
CREATE TABLE `t_products` (
  `product_id` int NOT NULL AUTO_INCREMENT COMMENT '产品ID',
  `product_name` varchar(100) NOT NULL COMMENT '产品名称',
  `category` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '产品类别',
  `unit_price` decimal(10,2) NOT NULL COMMENT '单价',
  `stock_quantity` int DEFAULT '0' COMMENT '库存数量',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`product_id`)
) ENGINE=InnoDB AUTO_INCREMENT=16 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='产品信息表';


INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(1, 'iPhone 15 Pro', '电子产品', 8999.00, 50, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(2, 'MacBook Air M2', '电子产品', 9499.00, 30, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(3, 'iPad Pro 12.9"', '电子产品', 8499.00, 40, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(4, 'AirPods Pro 2', '电子产品', 1899.00, 120, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(5, 'Dell XPS 13', '电子产品', 7999.00, 25, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(6, 'Nike Air Max 270', '服装', 899.00, 100, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(7, 'Adidas Superstar', '服装', 799.00, 90, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(8, 'Levi''s 501 牛仔裤', '服装', 699.00, 80, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(9, 'Uniqlo U系列 T恤', '服装', 199.00, 200, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(10, 'The North Face 冲锋衣', '服装', 1299.00, 60, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(11, '可口可乐 330ml', '食品', 3.50, 500, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(12, '百事可乐 330ml', '食品', 3.20, 450, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(13, '乐事薯片 原味', '食品', 7.80, 300, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(14, '奥利奥饼干', '食品', 12.50, 400, '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_products
(product_id, product_name, category, unit_price, stock_quantity, created_at, updated_at)
VALUES(15, '蒙牛纯牛奶 250ml', '食品', 3.80, 600, '2025-08-24 20:44:41', '2025-08-24 20:44:41');


-- t_sales_orders definition
DROP TABLE IF EXISTS `t_sales_orders`;
CREATE TABLE `t_sales_orders` (
  `order_id` int NOT NULL AUTO_INCREMENT COMMENT '订单ID',
  `order_number` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '订单编号',
  `customer_id` int NOT NULL COMMENT '客户ID',
  `order_date` date NOT NULL COMMENT '订单日期',
  `total_amount` decimal(12,2) NOT NULL COMMENT '订单总金额',
  `status` enum('Pending','Shipped','Delivered','Cancelled') DEFAULT 'Pending' COMMENT '订单状态',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`order_id`),
  UNIQUE KEY `order_number` (`order_number`),
  KEY `customer_id` (`customer_id`)
) ENGINE=InnoDB AUTO_INCREMENT=21 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='销售订单主表';

INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(1, 'SO20250820001', 1, '2025-08-20', 9022.40, 'Delivered', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(2, 'SO20250820002', 2, '2025-08-20', 9519.00, 'Delivered', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(3, 'SO20250821001', 3, '2025-08-21', 3805.60, 'Shipped', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(4, 'SO20250821002', 4, '2025-08-21', 811.80, 'Shipped', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(5, 'SO20250821003', 5, '2025-08-21', 8511.50, 'Shipped', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(6, 'SO20250822001', 6, '2025-08-22', 2447.00, 'Pending', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(7, 'SO20250822002', 7, '2025-08-22', 1010.60, 'Pending', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(8, 'SO20250822003', 8, '2025-08-22', 8024.60, 'Pending', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(9, 'SO20250823001', 9, '2025-08-23', 918.00, 'Shipped', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(10, 'SO20250823002', 10, '2025-08-23', 1341.00, 'Shipped', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(11, 'SO20250823003', 1, '2025-08-23', 8999.00, 'Shipped', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(12, 'SO20250823004', 2, '2025-08-23', 9499.00, 'Shipped', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(13, 'SO20250824001', 3, '2025-08-24', 1924.00, 'Pending', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(14, 'SO20250824002', 4, '2025-08-24', 1605.80, 'Pending', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(15, 'SO20250824003', 5, '2025-08-24', 8515.00, 'Pending', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(16, 'SO20250824004', 6, '2025-08-24', 678.00, 'Pending', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(17, 'SO20250824005', 7, '2025-08-24', 8010.40, 'Pending', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(18, 'SO20250824006', 8, '2025-08-24', 899.00, 'Cancelled', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(19, 'SO20250824007', 9, '2025-08-24', 609.50, 'Pending', '2025-08-24 20:44:41', '2025-08-24 20:44:41');
INSERT INTO t_sales_orders
(order_id, order_number, customer_id, order_date, total_amount, status, created_at, updated_at)
VALUES(20, 'SO20250824008', 10, '2025-08-24', 1314.60, 'Pending', '2025-08-24 20:44:41', '2025-08-24 20:44:41');

-- view_alarm_detail source

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









