# banksys_wjk — 银行营销数据交互分析与在线预测系统

基于 **Streamlit + scikit-learn + Docker** 的银行营销数据分析与认购预测 Web 应用。

## 功能

| 功能 | 说明 |
|------|------|
| 📊 **数据分析** | 交互式图表探索客户特征分布、订阅转化率、相关性热力图 |
| 🔮 **在线预测** | 基于离线训练的 ML 模型,通过点选表单输入客户特征,实时预测认购概率 |

## 技术栈

- **Python 3.11**
- **Streamlit** — Web 框架
- **pandas / numpy / plotly / seaborn** — 数据处理与可视化
- **scikit-learn** — 机器学习
- **pytest / ruff** — 测试与代码质量
- **Docker** — 容器化部署
- **GitHub Actions** — CI/CD

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt -r requirements-dev.txt

# 训练模型(离线)
python -m src.train --data-path data/train.csv --model-path models/model.joblib

# 启动应用(开发模式)
streamlit run app.py
```

## 端口

- Streamlit 默认端口: `8501`(容器内)
- 主机映射端口: `9333`

## 项目结构

```text
banksys_wjk/
├── app.py                  # Streamlit 主入口
├── pages/                  # 子页面
│   ├── 01_data_analysis.py
│   └── 02_prediction.py
├── src/                    # 核心逻辑
│   ├── data_loader.py
│   ├── analysis.py
│   ├── train.py
│   └── predict.py
├── data/                   # 数据文件
├── models/                 # 训练好的模型
├── tests/                  # 测试
├── standards/              # 项目规范
└── Dockerfile
```
