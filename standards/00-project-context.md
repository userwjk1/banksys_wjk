# 00 · 项目上下文 〔本项目活记忆 · AI 维护〕

> **作用**:这是项目的"身份档案"。AI 接管项目时先读这里,了解项目目标、技术栈、目录、部署取值。
> **更新时机**:架构、技术栈、目录结构、端口、部署目录、重要约束变化时更新。
> **填写方式**:把 `<...>` 替换成真实内容;用不到的行删掉。

---

## 1. 项目是什么

- **项目名称**: `banksys_wjk`
- **一句话目标**: 银行营销数据交互分析与在线预测系统 —— 支持数据可视化探索,并基于离线训练的 ML 模型提供在线认购预测
- **使用者/受益者**: 银行业务分析师、营销人员
- **核心功能**:
  - **功能一 · 数据分析交互页面**: 上传/加载银行营销数据,提供交互式图表(分布图、相关性热力图、转化漏斗等)、筛选、统计摘要,辅助业务决策
  - **功能二 · 在线预测系统**: 离线训练二分类模型(预测客户是否认购定期存款),保存模型;Web 页面以点选/下拉表单收集客户特征,实时返回预测结果与置信度
- **输入/数据**: 银行营销数据集(项目 `data/` 目录,含 `train.csv`(训练集,含目标列)与 `test.csv`(预测集,无目标列);典型字段:age、job、marital、education、default、housing、loan、contact、month、day_of_week、duration、campaign、pdays、previous、poutcome、emp_var_rate、cons_price_index、cons_conf_index、lending_rate3m、nr_employed;目标列为 `subscribe`(yes/no))

## 2. 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| 语言/运行时 | Python 3.11 | 教学常用版本,稳定 |
| Web/可视化框架 | Streamlit | 快速构建数据应用,内置交互组件,适合分析+预测页面 |
| 数据处理 | pandas、numpy | 数据分析基石 |
| 可视化 | plotly、matplotlib、seaborn | 交互式图表 + 静态图表;plotly 与 Streamlit 集成好 |
| 机器学习 | scikit-learn | 经典二分类模型(LogisticRegression/RandomForest),模型持久化(joblib/pickle) |
| 测试 | pytest | Python 标准测试框架 |
| 格式/静态检查 | ruff | 快速、现代化 |
| 打包/运行 | Docker | 标准容器化部署 |
| CI/CD | GitHub Actions | 通用、可视化、适合教学 |

## 3. 目录地图

```text
banksys_wjk/
├── standards/                    # AI 项目记忆与通用规范
├── data/                         # 原始数据文件(已就位)
│   ├── train.csv                 # 银行营销训练集(含目标列 subscribe)
│   └── test.csv                  # 银行营销预测集(无目标列)
├── app.py                        # Streamlit 主入口(多页面)
├── pages/
│   ├── 01_data_analysis.py       # 功能一:数据分析交互页面
│   └── 02_prediction.py          # 功能二:在线预测页面
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # 数据加载与预处理
│   ├── analysis.py               # 数据分析逻辑(统计、聚合)
│   ├── train.py                  # 模型训练脚本(离线)
│   └── predict.py                # 模型加载与预测
├── models/                       # 训练好的模型文件
│   └── model.joblib
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_analysis.py
│   ├── test_train.py
│   └── test_predict.py
├── requirements.txt              # 生产运行依赖
├── requirements-dev.txt          # 开发/CI 依赖
├── Dockerfile                    # 容器构建
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
└── README.md
```

> 新增目录前先更新本节,避免项目越做越散。

## 4. 质量门槛

| 类型 | 本项目标准 |
|---|---|
| 格式检查 | `ruff format --check .` |
| 静态检查 | `ruff check .` |
| 单元测试 | `pytest` |
| 覆盖率 | >=70% |
| 构建 | `docker build` 成功 |
| 业务/模型指标 | 模型 AUC >= 0.75、测试集准确率 >= 80% |
| 数据校验 | 关键字段非空率 >= 90%、目标列无泄漏 |

## 5. 不变约束

- 密钥、密码、私钥、Token **绝不写进代码或文档**,只进 GitHub Secrets / 环境变量。
- `main` 分支受保护,日常开发必须走 feature 分支 + PR。
- CI 红灯不合并。
- 模型训练是离线步骤,不可在预测页面实时训练。
- 预测页面输入必须校验范围,防止非法值导致模型异常。

## 6. 部署/CI 占位符取值

> `guides/` 和 workflow 里的通用占位符,在本项目里的真实值只写这里。

| 占位符 | 本项目取值 | 说明 |
|---|---|---|
| `<APP>` | `banksys_wjk` | 应用名/镜像名/容器名 |
| `<DEPLOY_DIR>` | `/opt/banksys_wjk` | 服务器部署目录 |
| `<PORT>` | `9333` | 服务端口 |
| `<PORT_MAX>` | `9340` | 端口回退上限 |
| `<PYVER>` | `3.11` | Python 版本 |
| `<HEALTHCHECK>` | `/_stcore/health` | Streamlit 内置健康检查端点 |
| `<SSH_USER>` | `root` | 部署用户(示例) |
| `<SSH_HOST>` | `<服务器IP>` | 服务器地址(由学生配置) |
