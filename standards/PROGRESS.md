# PROGRESS · banksys_wjk 〔本项目活记忆 · 状态机〕

> **作用**:这是项目的"存档点"。任意 AI、任意重启会话,读它即可知道当前做到哪、下一步做什么、踩过什么坑。
> **更新时机**:每完成一个有意义步骤、每次会话结束前。
> **格式要求**:时间倒序,最新在上;短、准、可接力。

---

## 当前状态 (最后更新: 2026-08-02 · by AI)

- **阶段**: `项目初始化 — 等待用户确认需求`
- **对应六步流程**: 第①步之前 — 填写项目上下文与需求文档
- **上一步完成**: 已读取 standards/README.md 及 02~06 规范;已填写 00-project-context.md 和 01-requirements.md
- **下一步 (TODO 第一条)**: 用户确认 00/01 文档内容无误后,进入六步流程第①步(建仓 + 配 Secrets)
- **阻塞项**: 
  - ⚠️ **等待用户确认** `00-project-context.md` 和 `01-requirements.md` 内容(已根据实际数据文件修正目标列名 `subscribe`、数据路径 `data/train.csv` + `data/test.csv`)。

---

## 待办清单 (TODO,按优先级)

- [ ] **用户确认** `00-project-context.md` 内容(技术栈、目录结构、端口、质量门槛)
- [ ] **用户确认** `01-requirements.md` 内容(6 个用户故事 + 验收标准,已修正目标列 `subscribe`、数据文件 `train.csv`/`test.csv`)
- [ ] 六步流程第①步:创建 GitHub 仓库 `banksys_wjk`(开源),初始化基础结构
- [ ] 六步流程第①步:提示用户配置 GitHub Secrets(`SSH_PRIVATE_KEY`/`SSH_HOST`/`SSH_USER`)
- [ ] 六步流程第②步:从 main 开 feature 分支 `feature/1-init-project`
- [ ] 实现 US-1:创建项目骨架(`app.py`、`pages/`、`src/`、`tests/`、`requirements.txt`、`requirements-dev.txt`、`Dockerfile`)
- [ ] 实现 US-2:数据加载与预处理模块(`src/data_loader.py`) + 测试
- [ ] 实现 US-3:数据分析交互页面(`pages/01_data_analysis.py`) + 测试
- [ ] 实现 US-4:模型离线训练脚本(`src/train.py`) + 测试 + 生成模型文件
- [ ] 实现 US-5:在线预测页面(`pages/02_prediction.py`) + 测试
- [ ] 实现 US-6:容器化部署验证(Dockerfile + docker build 成功)
- [ ] 本地 CI 自检:ruff + pytest + 覆盖率 >= 80% + 模型 AUC >= 0.75
- [ ] 创建 CI workflow (`.github/workflows/ci.yml`)
- [ ] 创建 CD workflow (`.github/workflows/cd.yml`)
- [ ] 推送 + 创建 PR,CI 全绿后人工合并,CD 自动部署到端口 9333
- [ ] 验证部署:访问 `http://<SSH_HOST>:9333` 确认两个功能页面正常

---

## 关键决策记录 (ADR)

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-08-02 | 选择 Streamlit 而非 Flask+前端框架 | 银行数据分析+预测场景需要快速构建交互式图表和表单,Streamlit 内置组件丰富,适合数据应用原型 |
| 2026-08-02 | 采用 sklearn Pipeline 封装预处理+模型 | 确保预测时预处理逻辑与训练时一致,避免训练/预测数据预处理代码重复 |
| 2026-08-02 | 训练排除 `duration` 特征 | `duration`(通话时长)在预测时未知,属数据泄漏;学术界和工业界共识:该特征不可用于真实预测 |
| 2026-08-02 | 容器内端口 8501 映射主机 9333 | Streamlit 默认 8501;用户要求外部端口 9333 |
| 2026-08-02 | 仓库开源 | 用户明确要求开源仓库 |

---

## 已知坑 (GOTCHAS)

- **数据格式已确认**:`data/train.csv`(含目标列 `subscribe`)和 `data/test.csv`(无目标列)已在项目内可直接使用。训练用 train.csv,预测演示用 test.csv。
- **Streamlit 健康检查端点**:Streamlit 内置健康检查路径为 `/_stcore/health`(非 `/health`),CD 部署脚本需使用正确路径。
- **模型文件需预训练**:镜像构建时模型文件必须已存在(离线训练后 COPY 进镜像),不可在容器启动时训练。

---

## 里程碑 (TODO)

- [ ] US-1 初始化项目工程化与 CI/CD
- [ ] US-2 数据加载与预处理模块
- [ ] US-3 数据分析交互页面
- [ ] US-4 模型离线训练
- [ ] US-5 在线预测页面
- [ ] US-6 容器化部署
