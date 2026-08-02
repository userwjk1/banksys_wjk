# 01 · 需求 / 活 PRD 〔本项目活记忆 · AI 维护〕

> **作用**:这是本项目唯一的需求文档。所有新功能、缺陷、技术债都追加到这里,不要另起多个 PRD 文件。
> **更新时机**:每次有新需求、需求变更、验收标准变化时更新。

---

## 1. 需求来源

| 类型 | 来源 | 进入方式 |
|---|---|---|
| 功能需求 Feature | 银行营销数据分析+在线预测 | 写成用户故事 |
| 缺陷 Bug | 测试 / CI/CD 失败 | 写复现步骤和期望结果 |
| 技术债 Tech Debt | Review / CI/CD 故障 | 写影响和修复目标 |

---

## 2. Issue 生命周期

| 阶段 | 状态 | 动作 |
|---|---|---|
| 提出 | Open | 写清场景、目标、验收标准 |
| 排期 | Backlog / Todo | 决定优先级和负责人 |
| 开发 | In Progress | 从 main 开 feature 分支 |
| 评审 | In Review | 提 PR,等待 CI 和 Review |
| 合并 | Done | PR 合并 main,自动关闭 Issue |
| 验收 | Verified | 按验收标准确认 |

**追踪规则**:分支名带 Issue 号,PR 描述写 `closes #<编号>`。

---

## 3. 用户故事模板

```text
### US-<编号> <一句话标题> · 状态: Backlog
作为 <角色>,
我想要 <能力>,
以便 <价值>。

验收标准:
- AC1: Given <前提>,When <动作>,Then <可验证结果>。
- AC2: <补充标准>

技术备注:
- <可选:约束、边界、风险>
```

---

## 4. 需求清单

### US-1 初始化项目工程化与 CI/CD · 状态: Backlog

作为 **项目开发者**,
我想要 项目具备基础工程结构、测试、CI 与 CD,
以便 后续每次开发都能自动检查并自动部署。

验收标准:
- AC1: 从 `main` 开 feature 分支完成初始化,不直接 push main。
- AC2: PR 触发 CI,至少包含 ruff 格式检查、ruff 静态检查、pytest 单元测试(覆盖率 >= 80%)、docker build 构建检查。
- AC3: CI 全绿后合并 main。
- AC4: 合并 main 自动触发 CD,部署到端口 9333(或回退区间 9333-8895),Streamlit 健康检查通过。
- AC5: 完成后更新 `standards/PROGRESS.md`。

技术备注:
- 容器名和应用名统一为 `banksys_wjk`。
- Streamlit 默认端口 8501,本容器映射到主机 9333。

---

### US-2 数据加载与预处理模块 · 状态: Backlog

作为 **数据分析师**,
我想要 系统能正确加载银行营销 CSV 数据并进行基础预处理,
以便 后续分析和建模使用的数据是干净、一致的。

验收标准:
- AC1: Given 银行营销数据 `data/train.csv` 和 `data/test.csv` 存在,When 调用 `load_data()` 函数,Then 返回非空 DataFrame,列名与原始数据一致(train 含 22 列含 subscribe,test 含 21 列无 subscribe)。
- AC2: Given 数据包含缺失值,When 预处理执行,Then 缺失值按策略处理(数值列填中位数/分类列填众数),并记录处理日志。
- AC3: Given 分类特征(如 job、marital、education 等),When 数据预处理,Then 自动完成 Label Encoding 或 One-Hot Encoding,编码后的特征矩阵可通过 `get_processed_data()` 获取。
- AC4: Given 数据路径不存在,When 加载数据,Then 抛出明确的 `FileNotFoundError` 并给出路径提示。
- AC5: 配套单元测试覆盖 ≥ 3 个场景(正常加载、缺失值处理、文件不存在)。

技术备注:
- 典型字段:id、age、job、marital、education、default、housing、loan、contact、month、day_of_week、duration、campaign、pdays、previous、poutcome、emp_var_rate、cons_price_index、cons_conf_index、lending_rate3m、nr_employed、subscribe(仅 train.csv)。
- 目标列 `subscribe` 取值 `yes`/`no`,需转为 1/0。
- train.csv 用于模型训练与评估;test.csv 不含目标列,用于在线预测演示/批量预测。
- 训练时不可使用 `duration` 列(该列在通话结束前未知),该列为事后信息,会导致数据泄漏。

---

### US-3 数据分析交互页面 · 状态: Backlog

作为 **银行业务分析师**,
我想要 通过一个交互式 Web 页面探索银行营销数据,
以便 直观了解客户特征分布、订阅转化情况,辅助营销决策。

验收标准:
- AC1: Given Streamlit 应用运行,When 访问数据分析页面,Then 页面显示数据总览(总行数、总列数、订阅率、各列数据类型)。
- AC2: Given 页面加载完成,When 用户选择某数值列(如 age),Then 显示该列的直方图/箱线图及描述性统计(均值、中位数、标准差、四分位数)。
- AC3: Given 页面加载完成,When 用户选择某分类列(如 job),Then 显示该列的柱状图及各分类的订阅转化率。
- AC4: Given 页面加载完成,When 用户查看"相关性分析",Then 显示数值特征的相关性热力图。
- AC5: Given 页面加载完成,When 用户调整筛选条件(如年龄范围、职业),Then 图表和数据摘要实时更新。
- AC6: Given 子页面切换,When 用户在侧边栏导航点击,Then 能自由切换"数据分析"和"在线预测"两个页面。

技术备注:
- 使用 Streamlit 的 `multipage` 或 `pages/` 目录自动发现多页面。
- 图表用 plotly 实现(支持悬停、缩放、下载)。
- 页面需处理大数据量时的性能(抽样/缓存 `@st.cache_data`)。

---

### US-4 模型离线训练 · 状态: Backlog

作为 **数据科学家**,
我想要 基于历史数据离线训练一个二分类 ML 模型来预测客户是否认购,
以便 该模型可以保存并加载到在线预测系统中使用。

验收标准:
- AC1: Given 预处理后的训练数据,When 执行 `train.py`,Then 完成训练并输出模型评估指标(AUC、准确率、精确率、召回率、F1)。
- AC2: Given 训练完成,When 模型保存,Then `models/model.joblib` 文件生成且包含完整的 sklearn pipeline(预处理+模型)。
- AC3: Given 训练完成,Then 模型 AUC ≥ 0.75 且测试集准确率 ≥ 80%。
- AC4: Given 训练数据中不包含 `duration` 列,Then 训练过程确认无数据泄漏(检查训练特征列表不包含 `duration`)。
- AC5: Given 训练脚本,Then 可通过命令行 `python -m src.train --data-path data/train.csv --model-path models/model.joblib` 运行。
- AC6: 配套单元测试覆盖:模型训练可运行、pipeline 结构正确、模型文件可保存和加载。

技术备注:
- 使用 `sklearn.pipeline.Pipeline` 封装预处理+模型,确保预测时预处理逻辑一致。
- 模型至少尝试 LogisticRegression 和 RandomForestClassifier,选 AUC 更好的保存。
- 数据划分用 `train_test_split` + `StratifiedKFold`(处理不平衡)。

---

### US-5 在线预测页面 · 状态: Backlog

作为 **银行营销人员**,
我想要 通过一个点选表单输入潜在客户的特征信息,
以便 即时获得该客户是否会认购定期存款的预测结果和置信度。

验收标准:
- AC1: Given 预测模型已训练并保存,When 访问预测页面,Then 显示一个表单,包含所有必要特征字段的输入控件(下拉框选分类值、滑块/数字框输入数值)。
- AC2: Given 用户填写完所有必填字段,When 点击"开始预测"按钮,Then 页面显示预测结果(会认购 / 不会认购)以及预测概率/置信度。
- AC3: Given 用户输入的数值超出合理范围(如 age=500),Then 表单在前端或后端校验并给出友好错误提示,不调用模型。
- AC4: Given 模型文件不存在,When 预测页面加载,Then 显示明确提示"模型尚未训练,请先运行训练脚本"。
- AC5: Given 预测功能,When 输入同一组特征多次,Then 每次预测结果一致(模型确定性推理)。
- AC6: 配套单元测试覆盖:正常预测、输入校验、模型文件缺失。

技术备注:
- 表单控件选型:
  - 数值特征(age、campaign、pdays 等):`st.slider` 或 `st.number_input`,限制合理范围。
  - 分类特征(job、marital、education 等):`st.selectbox`,选项从训练数据取值中提取。
- 预测输入不包含 `duration` 列(与训练一致)。
- 加载模型使用 `@st.cache_resource` 缓存,避免每次预测都重新加载。

---

### US-6 容器化部署 · 状态: Backlog

作为 **运维工程师**,
我想要 Streamlit 应用以 Docker 容器运行,
以便 统一部署环境并支持 CI/CD 自动部署。

验收标准:
- AC1: Given Dockerfile 已就绪,When 执行 `docker build -t banksys_wjk:latest .`,Then 构建成功无错误。
- AC2: Given 镜像已构建,When 执行 `docker run -d -p 9333:8501 --name banksys_wjk banksys_wjk:latest`,Then 容器正常启动。
- AC3: Given 容器已运行,Then Streamlit 应用可通过 `http://localhost:9333` 访问,含数据分析页和预测页两个页面。
- AC4: Given 部署完成,Then Streamlit 内置健康检查端点 `/_stcore/health` 返回 200。
- AC5: Dockerfile 支持国内镜像源参数 `PIP_INDEX_URL`,默认使用清华源。

技术备注:
- 基础镜像 `python:3.11-slim`。
- Streamlit 容器内默认端口 8501,映射到主机 9333。
- 数据文件通过 `COPY` 或 volume 挂载进入容器(小数据直接 COPY,大数据用 volume)。
- 模型文件 `models/model.joblib` 需在构建镜像前训练好并 COPY 进镜像。

---

## 5. 非功能需求

- **安全**:密钥只进 Secrets,不进 Git;预测输入校验防止恶意输入。
- **可维护**:一需求一小 PR,避免大爆炸式提交。
- **可测试**:核心逻辑必须有单元测试,覆盖率 >= 80%;数据分析函数和预测函数均为纯函数,易于测试。
- **可部署**:部署后 Streamlit 健康检查必须返回 200;端口 9333。
- **代码质量**:通过 ruff 格式检查和静态检查。
- **模型质量**:AUC ≥ 0.75,准确率 ≥ 80%,训练过程无数据泄漏。
- **性能**:数据分析页面首次加载 ≤ 5 秒(带缓存);单次预测响应 ≤ 2 秒。
