# 借贷APP用户画像数仓设计方案

## 1. 概述与设计原则

本次数仓设计旨在为拥有 **5000万注册用户** 和 **300万交易用户** 的借贷APP构建一个高效、稳定、可扩展的用户画像系统。设计遵循经典的 **4层架构**，并结合业务特点，采用 **“离线全量 + 实时增量”** 的混合更新策略，以实现成本与实时性的最佳平衡。

### 核心设计原则

1.  **分层处理（Layering）**: 采用 ODS、DWD、DWS、ADS 四层结构，实现数据解耦、提高复用性。
2.  **分群策略（Segmentation）**: 对 5000万全量用户进行离线 T+1 更新，对 300万核心交易用户进行准实时更新，确保核心业务（风控、放款）的实时性要求。
3.  **技术选型（Technology）**: 离线采用 **Hive/Spark** 进行批处理，实时采用 **Kafka/Flink** 进行流处理，存储结合 **HDFS/Parquet** (离线) 和 **Redis/ClickHouse** (实时/OLAP)。

## 2. 数仓分层结构与内容

数仓采用经典的 **ODS (Operational Data Store) -> DWD (Data Warehouse Detail) -> DWS (Data Warehouse Summary) -> ADS (Application Data Store)** 四层结构。

| 层级 | 名称 | 作用与目标 | 主要内容 |
| :--- | :--- | :--- | :--- |
| **ODS** | 原始数据层 | 保持数据原貌，不做任何清洗和转换，是数据仓库的起点。 | 业务系统（MySQL/PostgreSQL）全量/增量快照；用户行为日志（Kafka/Flume）原始数据。 |
| **DWD** | 数据明细层 | 进行数据清洗、规范化、维度退化，构建一致性的事实表和维度表。 | **事实表**: 用户注册、登录、借款申请、放款、还款、逾期事件明细。**维度表**: 用户信息、产品信息、渠道信息。 |
| **DWS** | 数据汇总层 | 基于DWD层数据，按主题、业务过程进行轻度或中度聚合，沉淀公共指标。 | **主题域**: 用户行为（登录、访问）、借贷交易（成功/失败次数、金额）、账户状态（会员、黑名单）。 |
| **ADS** | 应用数据层 | 面向具体应用场景（如用户画像、BI报表、风控系统）构建的数据集。 | **ads_user_base_profile** (5000w 离线全量画像)，**ads_user_risk_profile** (300w 实时风险画像)。 |

## 3. DWD层表结构设计（部分核心表）

DWD层是数据清洗和标准化的核心，我们基于用户画像字段和业务流程，设计以下核心事实表和维度表。

### DWD 维度表：dwd_dim_user_info (用户基础信息)

用于存储用户静态和慢变维度信息。

| 字段名 | 字段说明 | 来源ODS表 | 备注 |
| :--- | :--- | :--- | :--- |
| user_no | 用户编号 | ods_user_base | **主键**，用户唯一标识 |
| register_time | 注册时间 | ods_user_base | |
| age | 年龄 | ods_user_base | |
| education | 学历 | ods_user_base | 规范化：本科/专科/高中... |
| marital_status | 婚姻状况 | ods_user_base | 规范化：已婚/未婚/离异 |
| occupation | 职业 | ods_user_base | 规范化：公司员工/私营业主... |
| id_card_address | 身份证地址 | ods_user_base | |
| start_date | 生效日期 | - | 维度拉链表起始日期 |
| end_date | 失效日期 | - | 维度拉链表结束日期 |

### DWD 事实表：dwd_fact_loan_transaction (借款交易明细)

用于存储每一笔借款的完整生命周期事件。

| 字段名 | 字段说明 | 来源ODS表 | 备注 |
| :--- | :--- | :--- | :--- |
| transaction_id | 交易ID | ods_loan_order | |
| user_no | 用户编号 | ods_loan_order | |
| apply_time | 申请时间 | ods_loan_order | |
| loan_amount | 申请金额 | ods_loan_order | |
| credit_limit | 授信额度 | ods_user_credit | |
| loan_status | 借款状态 | ods_loan_order | 0-申请中, 1-成功放款, 2-拒绝 |
| is_overdue | 是否逾期 | ods_loan_repay | 1-是, 0-否 |
| overdue_days | 逾期天数 | ods_loan_repay | 0表示未逾期 |
| repay_time | 实际还款时间 | ods_loan_repay | |

### DWD 事实表：dwd_fact_user_action (用户行为明细)

用于存储用户在APP内的关键行为日志。

| 字段名 | 字段说明 | 来源ODS表 | 备注 |
| :--- | :--- | :--- | :--- |
| log_id | 日志ID | ods_app_log | |
| user_no | 用户编号 | ods_app_log | |
| event_time | 事件时间 | ods_app_log | |
| event_name | 事件名称 | ods_app_log | 登录、点击、提交工单等 |
| page_id | 页面ID | ods_app_log | |
| work_order_id | 工单ID | ods_work_order | 仅工单事件有值 |

## 4. DWS层表结构设计（部分核心表）

DWS层主要进行主题聚合，为ADS层提供预计算的指标。

### DWS 聚合表：dws_user_loan_agg_td (用户历史借贷聚合)

聚合用户从注册到T-1日的所有借贷历史数据。

| 字段名 | 字段说明 | 来源DWD表 | 备注 |
| :--- | :--- | :--- | :--- |
| user_no | 用户编号 | dwd_fact_loan_transaction | **主键** |
| total_loan_amount | 累计借款总金额 | dwd_fact_loan_transaction | SUM(loan_amount) |
| successful_loans | 历史成功贷款次数 | dwd_fact_loan_transaction | COUNT(CASE WHEN loan_status = 1) |
| overdue_loan_count | 历史逾期贷款次数 | dwd_fact_loan_transaction | COUNT(CASE WHEN is_overdue = 1) |
| normal_loan_count | 历史正常贷款次数 | dwd_fact_loan_transaction | successful_loans - overdue_loan_count |
| last_loan_time | 最近一次放款时间 | dwd_fact_loan_transaction | MAX(apply_time WHERE loan_status = 1) |
| dt | 分区日期 | - | T-1日期分区 |

### DWS 聚合表：dws_user_action_td (用户历史行为聚合)

聚合用户从注册到T-1日的所有行为数据。

| 字段名 | 字段说明 | 来源DWD表 | 备注 |
| :--- | :--- | :--- | :--- |
| user_no | 用户编号 | dwd_fact_user_action | **主键** |
| last_login_time | 最后登录时间 | dwd_fact_user_action | MAX(event_time WHERE event_name = 'login') |
| work_order_count | 累计工单数量 | dwd_fact_user_action | COUNT(CASE WHEN event_name = 'submit_work_order') |
| recent_chat_count | 近期聊天次数 | dwd_fact_user_action | COUNT(CASE WHEN event_name = 'chat_with_cs' AND event_time >= DATE_SUB(dt, 30)) |
| dt | 分区日期 | - | T-1日期分区 |

## 5. ADS层用户画像表结构

ADS层是最终面向业务的画像表，我们根据用户画像需求，将数据整合为一张宽表。

### ADS 应用表：ads_user_profile_td (用户画像宽表)

这张表将整合所有离线计算的画像字段，用于BI分析和离线营销。

| 类别 | 字段名 | 字段说明 | 来源DWD/DWS表 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **基本信息** | user_no | 用户编号 | dwd_dim_user_info | **主键** |
| | register_time | 注册时间 | dwd_dim_user_info | |
| | last_login_time | 最后登录时间 | dws_user_action_td | |
| | age | 年龄 | dwd_dim_user_info | |
| | education | 学历 | dwd_dim_user_info | |
| | marital_status | 婚姻状况 | dwd_dim_user_info | |
| | occupation | 职业 | dwd_dim_user_info | |
| | id_card_address | 身份证地址 | dwd_dim_user_info | |
| **财务信息** | credit_limit | 信用额度 | dwd_fact_loan_transaction | 取最新值 |
| | loan_amount | 贷款金额 | dws_user_loan_agg_td | 当前未结清余额（需DWS层计算） |
| | income_range | 收入范围 | dwd_dim_user_info | 需从用户认证信息中提取 |
| | successful_loans | 成功贷款次数 | dws_user_loan_agg_td | |
| | overdue_loan_count | 逾期贷款次数 | dws_user_loan_agg_td | |
| | normal_loan_count | 正常贷款次数 | dws_user_loan_agg_td | |
| **账户状态** | is_member | 是否会员 | dwd_dim_user_info | |
| | is_renewal | 是否续费 | dwd_dim_user_info | |
| | account_status | 账户状态 | dwd_dim_user_info | |
| | is_blacklisted | 是否黑名单 | dwd_dim_user_info | |
| **行为数据** | work_order_count | 工单数量 | dws_user_action_td | |
| | recent_chat_count | 最近聊天次数 | dws_user_action_td | |
| **分区字段** | dt | 分区日期 | - | T-1日期分区 |

## 6. 离线全量计算（Hive SQL）

### 6.1. DWS层：dws_user_loan_agg_td 计算

从DWD层借款交易明细表 `dwd_fact_loan_transaction` 聚合用户历史借贷指标。

```sql
-- 假设使用 Hive/Spark SQL，dt 为 T-1 日期分区
INSERT OVERWRITE TABLE dws_user_loan_agg_td PARTITION (dt = '${biz_date}')
SELECT
    t1.user_no,
    SUM(t1.loan_amount) AS total_loan_amount,
    SUM(CASE WHEN t1.loan_status = 1 THEN 1 ELSE 0 END) AS successful_loans,
    SUM(CASE WHEN t1.is_overdue = 1 THEN 1 ELSE 0 END) AS overdue_loan_count,
    SUM(CASE WHEN t1.loan_status = 1 AND t1.is_overdue = 0 THEN 1 ELSE 0 END) AS normal_loan_count,
    MAX(CASE WHEN t1.loan_status = 1 THEN t1.apply_time ELSE NULL END) AS last_loan_time
FROM
    dwd_fact_loan_transaction t1
WHERE
    t1.dt <= '${biz_date}' -- 聚合历史所有数据
GROUP BY
    t1.user_no;
```

### 6.2. DWS层：dws_user_action_td 计算

从DWD层用户行为明细表 `dwd_fact_user_action` 聚合用户历史行为指标。

```sql
-- 假设使用 Hive/Spark SQL，dt 为 T-1 日期分区
INSERT OVERWRITE TABLE dws_user_action_td PARTITION (dt = '${biz_date}')
SELECT
    t1.user_no,
    MAX(CASE WHEN t1.event_name = 'login' THEN t1.event_time ELSE NULL END) AS last_login_time,
    SUM(CASE WHEN t1.event_name = 'submit_work_order' THEN 1 ELSE 0 END) AS work_order_count,
    SUM(CASE WHEN t1.event_name = 'chat_with_cs' AND t1.event_time >= DATE_SUB('${biz_date}', 30) THEN 1 ELSE 0 END) AS recent_chat_count
FROM
    dwd_fact_user_action t1
WHERE
    t1.dt <= '${biz_date}' -- 聚合历史所有数据
GROUP BY
    t1.user_no;
```

### 6.3. ADS层：ads_user_profile_td 最终画像宽表计算

将DWD维度表和DWS聚合表进行关联，生成最终的离线用户画像宽表。

```sql
-- 假设使用 Hive/Spark SQL，dt 为 T-1 日期分区
INSERT OVERWRITE TABLE ads_user_profile_td PARTITION (dt = '${biz_date}')
SELECT
    -- 基本信息字段 (来自 DWD 维度表)
    t1.user_no,
    t1.register_time,
    t2.last_login_time,
    t1.age,
    t1.education,
    t1.marital_status,
    t1.occupation,
    t1.id_card_address,

    -- 财务信息字段 (来自 DWD 维度表和 DWS 聚合表)
    t3.credit_limit, -- 假设 t3 已经计算出最新的授信额度
    t4.current_loan_balance AS loan_amount, -- 假设 DWS 层计算出当前未结清余额
    t1.income_range, -- 假设 DWD 维度表已包含
    t2.successful_loans,
    t2.overdue_loan_count,
    t2.normal_loan_count,

    -- 账户状态字段 (来自 DWD 维度表)
    t1.is_member,
    t1.is_renewal,
    t1.account_status,
    t1.is_blacklisted,

    -- 行为数据字段 (来自 DWS 聚合表)
    t3.work_order_count,
    t3.recent_chat_count
FROM
    dwd_dim_user_info t1 -- 用户基础信息 (假设是拉链表，取最新有效记录)
LEFT JOIN
    dws_user_loan_agg_td t2 ON t1.user_no = t2.user_no AND t2.dt = '${biz_date}'
LEFT JOIN
    dws_user_action_td t3 ON t1.user_no = t3.user_no AND t3.dt = '${biz_date}'
LEFT JOIN
    (SELECT user_no, credit_limit, income_range, is_member, is_renewal, account_status, is_blacklisted FROM dwd_dim_user_info WHERE end_date = '9999-12-31') t4 ON t1.user_no = t4.user_no
LEFT JOIN
    (SELECT user_no, SUM(loan_amount - repaid_amount) AS current_loan_balance FROM dwd_fact_loan_transaction WHERE loan_status = 1 AND is_fully_repaid = 0 GROUP BY user_no) t5 ON t1.user_no = t5.user_no;
```

## 7. 实时增量方案（Kafka + Flink + Redis）

实时增量方案主要针对 **300万交易用户** 的 **关键风控状态** 字段，以满足贷前、贷中决策的秒级响应要求。

### 7.1. 实时更新字段

| 字段类别 | 字段名 | 实时性要求 | 存储介质 |
| :--- | :--- | :--- | :--- |
| **财务信息** | loan_amount (当前未结清余额) | 准实时 | Redis/HBase |
| **账户状态** | is_blacklisted (是否黑名单) | 秒级 | Redis |
| **风控状态** | current_overdue_days (当前逾期天数) | 秒级 | Redis |

### 7.2. 实时链路设计

1.  **数据源 (Kafka)**: 业务系统（如订单系统、风控系统）将关键事件（放款成功、还款成功、逾期发生、黑名单变更）实时写入 Kafka Topic。
2.  **计算引擎 (Flink)**: Flink 消费 Kafka 数据流，进行实时计算和状态维护。
3.  **存储 (Redis)**: Flink 将计算结果写入 Redis，供风控系统通过 API 实时查询。

### 7.3. Flink 实时计算伪代码

我们以 **实时更新 `loan_amount` (当前未结清余额)** 为例。

**输入 Kafka Topic: `loan_transaction_events`**

消息格式 (JSON):
```json
{
  "user_no": "100001",
  "event_type": "LOAN_SUCCESS" | "REPAY_SUCCESS" | "REPAY_PARTIAL",
  "amount": 20000,
  "transaction_id": "TX20250101001"
}
```

**Flink SQL/DataStream 伪代码**

```sql
-- Flink SQL 示例：实时计算用户当前未结清余额
-- 1. 定义 Kafka Source Table
CREATE TABLE loan_events (
    user_no STRING,
    event_type STRING,
    amount DECIMAL(10, 2),
    transaction_id STRING,
    event_time TIMESTAMP(3),
    WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'loan_transaction_events',
    'properties.bootstrap.servers' = '...',
    'format' = 'json'
);

-- 2. 定义 Redis Sink Table (假设 Redis 存储 user_no -> current_loan_amount)
CREATE TABLE redis_loan_balance (
    user_no STRING,
    current_loan_amount DECIMAL(10, 2)
) WITH (
    'connector' = 'redis',
    'host' = '...',
    'key.field' = 'user_no',
    'value.field' = 'current_loan_amount'
);

-- 3. Flink DataStream 逻辑 (使用状态编程)
-- 实际生产中，需要使用 Flink DataStream API 的 Keyed State 来维护每个用户的余额状态。

-- 伪代码逻辑：
-- Stream<LoanEvent> events = env.fromSource(kafkaSource);
-- events
--     .keyBy(event -> event.user_no)
--     .process(new BalanceUpdateFunction()) // 核心状态维护逻辑
--     .addSink(new RedisSink());

-- BalanceUpdateFunction 核心逻辑:
--   - 维护 ValueState<Decimal> currentBalance
--   - 收到 LOAN_SUCCESS: currentBalance += amount
--   - 收到 REPAY_SUCCESS/REPAY_PARTIAL: currentBalance -= amount
--   - 将更新后的 currentBalance 写入 Redis
```

## 8. 总结与交付物

本次设计提供了完整的4层数仓架构，并针对借贷APP的业务特点，实现了：

1.  **离线全量画像**：覆盖 5000万用户，通过 Hive/Spark SQL 实现 T+1 批处理，确保数据准确性和成本效益。
2.  **实时增量画像**：覆盖 300万核心交易用户，通过 Kafka -> Flink -> Redis 链路实现秒级更新，满足风控和放款的实时决策需求。

交付物包括：
1.  数仓分层结构与内容表。
2.  DWD/DWS/ADS 核心表结构定义。
3.  离线全量计算的 Hive SQL 模版。
4.  实时增量计算的 Flink 伪代码方案。
