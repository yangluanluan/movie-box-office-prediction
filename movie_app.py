# 依赖安装（首次运行执行）
# pip install streamlit pandas numpy scikit-learn joblib plotly

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# -------------------------- 全局缓存工具（解决重复读取卡顿核心） --------------------------
# 缓存原始CSV，全局仅读取1次，页面刷新/点击预测不再重复IO
@st.cache_data
def load_raw_csv(csv_path="电影数据.csv"):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df

# 缓存下拉框演员、导演列表，不用每次读取后去重排序
@st.cache_data
def get_select_options(df):
    directors = sorted(df['DIRECTOR'].dropna().unique().tolist())
    actors1 = sorted(df['ACTOR 1'].dropna().unique().tolist())
    actors2 = sorted(df['ACTOR 2'].dropna().unique().tolist())
    return directors, actors1, actors2

# 缓存模型实例，全局只加载一次模型、编码器
@st.cache_resource
def get_predictor_instance():
    pred = MovieBoxOfficePredictor()
    pred.load_models()
    return pred

# ===================== 模型预测类 =====================
class MovieBoxOfficePredictor:
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        self.model_scores = {}
        self.stacking_model = None
        self.full_oof = None
        self.feature_columns = None
        self.le = None
        self.mlb = None
        self.stacking_test_rmse = None
        self.stacking_test_r2 = None  # 新增保存Stacking的R²

    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def train_with_kfold(self, X, y, n_splits=5, random_state=42):
        self.feature_columns = list(X.columns)
        X_arr = X.values
        y_arr = y.values
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        results = {}
        oof_list = []

        for model_name, model in self.models.items():
            cv_rmse_scores = []
            cv_r2_scores = []  # 新增：保存每折R²
            oof_train = np.zeros(len(X_arr))
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_arr)):
                X_train = X_arr[train_idx]
                X_val = X_arr[val_idx]
                y_train = y_arr[train_idx]
                y_val = y_arr[val_idx]
                model_fold = model.__class__(**model.get_params())
                model_fit = model_fold.fit(X_train, y_train)
                val_pred = model_fit.predict(X_val)
                oof_train[val_idx] = val_pred
                # 同时计算RMSE、R²
                fold_rmse = self.rmse(y_val, val_pred)
                fold_r2 = r2_score(y_val, val_pred)
                cv_rmse_scores.append(fold_rmse)
                cv_r2_scores.append(fold_r2)

            results[model_name] = {
                'cv_rmse_scores': cv_rmse_scores,
                'cv_r2_scores': cv_r2_scores,
                'oof_train': oof_train,
                'mean_cv_rmse': np.mean(cv_rmse_scores),
                'std_cv_rmse': np.std(cv_rmse_scores),
                'mean_cv_r2': np.mean(cv_r2_scores)  # 保存平均R²
            }
            oof_list.append(oof_train)
            full_model = model.__class__(**model.get_params())
            full_model.fit(X_arr, y_arr)
            self.trained_models[model_name] = full_model

        self.full_oof = np.column_stack(oof_list)
        self.model_scores = results
        return results

    def train_stacking(self, X, y, test_size=0.2):
        if self.full_oof is None:
            raise Exception("请先训练基模型！")
        y_arr = y.values
        X_train, X_test, y_train, y_test = train_test_split(
            self.full_oof, y_arr, test_size=test_size, random_state=42
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_train_final = np.zeros(len(X_train))

        for train_idx, val_idx in kf.split(X_train):
            X_fold = X_train[train_idx]
            y_fold = y_train[train_idx]
            stacking_fold = RandomForestRegressor(
                n_estimators=100, random_state=42, max_depth=32, min_samples_split=2
            )
            stacking_fold.fit(X_fold, y_fold)
            oof_train_final[val_idx] = stacking_fold.predict(X_train[val_idx])

        self.stacking_model = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=32, min_samples_split=2
        )
        self.stacking_model.fit(X_train, y_train)

        stacking_pred = self.stacking_model.predict(X_test)
        stacking_rmse = self.rmse(y_test, stacking_pred)
        stacking_r2 = r2_score(y_test, stacking_pred)
        self.stacking_test_rmse = stacking_rmse
        self.stacking_test_r2 = stacking_r2  # 持久化保存Stacking的R²
        print(f"\n=== Stacking 堆叠模型训练完成 ===")
        print(f"Stacking RMSE: {stacking_rmse:.4f}")
        print(f"Stacking R²: {stacking_r2:.4f}")

    def predict_by_model(self, X, model_name):
        if not self.trained_models:
            raise Exception("模型未加载！")

        if model_name != "stacking_model":
            if model_name not in self.trained_models:
                raise Exception(f"模型 {model_name} 不存在！")
            return self.trained_models[model_name].predict(X)

        base_preds = []
        for m in self.models.keys():
            pred = self.trained_models[m].predict(X)
            base_preds.append(pred)
        stack_input = np.column_stack(base_preds)

        if self.stacking_model is None:
            raise Exception("Stacking堆叠模型未加载！")
        return self.stacking_model.predict(stack_input)

    def save_models(self):
        for model_name, model in self.trained_models.items():
            joblib.dump(model, f"{model_name}.pkl")
        if self.stacking_model is not None:
            joblib.dump(self.stacking_model, "stacking_model.pkl")
        if self.feature_columns:
            joblib.dump(self.feature_columns, "feature_columns.pkl")
        if self.le:
            joblib.dump(self.le, "label_encoder.pkl")
        if self.mlb:
            joblib.dump(self.mlb, "multi_label_binarizer.pkl")
        joblib.dump(self.stacking_test_rmse, "stacking_rmse.pkl")
        joblib.dump(self.stacking_test_r2, "stacking_r2.pkl")  # 保存Stacking的R²
        joblib.dump(self.model_scores, "model_scores.pkl")
        print("\n✅ 所有模型、性能指标(RMSE+R²)和特征配置已保存")

    def load_models(self):
        self.trained_models = {}
        for model_name in self.models.keys():
            file_path = f"{model_name}.pkl"
            if os.path.exists(file_path):
                self.trained_models[model_name] = joblib.load(file_path)
        stack_path = "stacking_model.pkl"
        if os.path.exists(stack_path):
            self.stacking_model = joblib.load(stack_path)
            print("✅ Stacking堆叠模型加载成功")
        rmse_file = "stacking_rmse.pkl"
        if os.path.exists(rmse_file):
            self.stacking_test_rmse = joblib.load(rmse_file)
        # 加载Stacking的R²
        r2_file = "stacking_r2.pkl"
        if os.path.exists(r2_file):
            self.stacking_test_r2 = joblib.load(r2_file)
        scores_path = "model_scores.pkl"
        if os.path.exists(scores_path):
            self.model_scores = joblib.load(scores_path)
            print("✅ 模型交叉验证性能指标(RMSE+R²)加载成功")
        if os.path.exists("feature_columns.pkl"):
            self.feature_columns = joblib.load("feature_columns.pkl")
        if os.path.exists("label_encoder.pkl"):
            self.le = joblib.load("label_encoder.pkl")
        if os.path.exists("multi_label_binarizer.pkl"):
            self.mlb = joblib.load("multi_label_binarizer.pkl")
        print("✅ 所有基模型和特征配置加载完成")

# ===================== 数据预处理函数（已修复：每个分类特征独立编码器，避免覆盖分级类别） =====================
def parse_gross(gross_str):
    if pd.isna(gross_str) or str(gross_str).strip() == "":
        return np.nan
    s = str(gross_str).replace('$', '').strip()
    if 'M' in s:
        return float(s.replace('M', '')) * 1_000_000
    elif 'K' in s:
        return float(s.replace('K', '')) * 1_000
    else:
        return float(s)

def extract_year(year_str):
    if pd.isna(year_str):
        return np.nan
    res = re.findall(r'\d{4}', str(year_str))
    return int(res[0]) if res else np.nan

def load_and_preprocess(csv_path):
    df = load_raw_csv(csv_path)
    df['GROSS COLLECTION'] = df['GROSS COLLECTION'].apply(parse_gross)
    df = df.dropna(subset=['GROSS COLLECTION']).reset_index(drop=True)

    feature_cols = [
        'Year', 'runtime', 'certificate', 'genre', 'RATING', 'metascore', 'votes',
        'DIRECTOR', 'ACTOR 1', 'ACTOR 2'
    ]
    X = df[feature_cols].copy()
    y_raw = df['GROSS COLLECTION']
    y = np.log1p(y_raw)

    X['Year'] = X['Year'].apply(extract_year)
    X['runtime'] = X.astype(str)['runtime'].str.replace(' min', '').astype(float)
    X['votes'] = X.astype(str)['votes'].str.replace(',', '').astype(float)

    X['DIRECTOR'] = X['DIRECTOR'].fillna("未知导演")
    X['ACTOR 1'] = X['ACTOR 1'].fillna("未知演员")
    X['ACTOR 2'] = X['ACTOR 2'].fillna("未知演员")

    drop_cols = ['Year', 'runtime', 'RATING', 'metascore', 'votes']
    combined = pd.concat([X, y], axis=1)
    combined = combined.dropna(subset=drop_cols).reset_index(drop=True)
    X = combined[feature_cols].copy()
    y = combined['GROSS COLLECTION']

    X['certificate'] = X['certificate'].fillna("unknown")

    # 每个分类特征单独创建LabelEncoder，互不覆盖
    le_certificate = LabelEncoder()
    X['certificate'] = le_certificate.fit_transform(X['certificate'].astype(str))

    le_director = LabelEncoder()
    X['DIRECTOR'] = le_director.fit_transform(X['DIRECTOR'].astype(str))

    le_actor1 = LabelEncoder()
    X['ACTOR 1'] = le_actor1.fit_transform(X['ACTOR 1'].astype(str))

    le_actor2 = LabelEncoder()
    X['ACTOR 2'] = le_actor2.fit_transform(X['ACTOR 2'].astype(str))

    def split_genre(x):
        return str(x).split(',')
    X['genre'] = X['genre'].apply(split_genre)
    mlb = MultiLabelBinarizer()
    genre_arr = mlb.fit_transform(X['genre'])
    genre_df = pd.DataFrame(genre_arr, columns=mlb.classes_)

    X = pd.concat([X.drop('genre', axis=1), genre_df], axis=1)
    X = X.astype(float)
    print(f"原始数据集总样本量：{len(df)}")
    print(f"清洗后可用有效样本量：{X.shape[0]}")

    # 只返回分级的编码器，用于前端下拉框展示所有BBFC分级
    return X, y, le_certificate, mlb

# ===================== Streamlit 网页应用 =====================
def run_streamlit_app(predictor):
    st.set_page_config(page_title="电影票房预测系统", layout="wide")
    st.markdown("""
        <style>
        .stApp {background-color: #f5f7fa;}
        .header {background-color: #6495ED; padding: 10px; color: white; font-size: 20px; font-weight: bold;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header">基于Python的电影票房分析预测系统</div>', unsafe_allow_html=True)
    menu = ["首页", "票房分析", "票房预测"]
    choice = st.sidebar.selectbox("导航", menu)

    # 首页
    if choice == "首页":
        st.title("基于Python的电影票房分析预测系统")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("系统介绍")
            st.write("""
            票房作为衡量电影能否盈利的重要指标，受诸多因素共同作用影响，
            且其影响机制较为复杂，票房的准确预测难度较大。本项目首先将影响电影类型、上映档期、导演、演员等因素量化处理并进行可视化分析。再利用多种回归模型与Stacking融合模型实现电影票房预测。
            """)

        with col2:
            st.info("本系统已取消登录验证，可直接前往【票房预测】或【票房分析】使用功能")

    # 票房分析页
    elif choice == "票房分析":
        st.title("📊 电影票房数据分析")
        st.write("以下是电影数据的可视化分析图表，帮助你理解票房的影响因素")
        st.divider()

        # 复用缓存CSV，不再重复读取硬盘
        df_raw = load_raw_csv()

        # 1. 数据集前10行展示
        st.subheader("一、原始数据集预览（前10行）")
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.caption(f"数据集总记录数：{len(df_raw)} 条")
        st.divider()

        # 2. 字段含义解释表格
        st.subheader("二、数据字段详细说明")
        field_info = [
            {"字段名": "Year", "中文名称": "上映年份", "字段含义": "电影正式上映的年份"},
            {"字段名": "runtime", "中文名称": "影片时长", "字段含义": "电影正片时长，单位：分钟"},
            {"字段名": "certificate", "中文名称": "影片分级", "字段含义": "英国BBFC电影分级，用于划分观影人群"},
            {"字段名": "genre", "中文名称": "电影类型", "字段含义": "影片题材类型，支持多类型组合，如动作、喜剧、科幻等"},
            {"字段名": "RATING", "中文名称": "大众评分", "字段含义": "普通观众给出的综合评分，满分10分"},
            {"字段名": "metascore", "中文名称": "专业影评分数", "字段含义": "专业影评人综合打分，满分100分"},
            {"字段名": "votes", "中文名称": "评价人数", "字段含义": "参与评分、投票的用户总数量"},
            {"字段名": "DIRECTOR", "中文名称": "导演", "字段含义": "电影主创导演姓名"},
            {"字段名": "ACTOR 1", "中文名称": "主演1", "字段含义": "第一主演/主要演员"},
            {"字段名": "ACTOR 2", "中文名称": "主演2", "字段含义": "第二主演/联合主演"},
            {"字段名": "GROSS COLLECTION", "中文名称": "总票房", "字段含义": "电影全球总票房，原始单位为美元，含M（百万）、K（千）单位标识"}
        ]
        field_df = pd.DataFrame(field_info)
        st.dataframe(field_df, use_container_width=True, hide_index=True)
        st.divider()

        # 数据清洗与转换（用于图表分析）
        df = df_raw.copy()
        df['GROSS COLLECTION'] = df['GROSS COLLECTION'].apply(parse_gross)
        df = df.dropna(subset=['GROSS COLLECTION']).reset_index(drop=True)
        df['Year'] = df['Year'].apply(extract_year)
        df['runtime'] = df.astype(str)['runtime'].str.replace(' min', '').astype(float)
        df['votes'] = df.astype(str)['votes'].str.replace(',', '').astype(float)

        # ===================== 新增：目标变量（票房）分布分析 =====================
        st.subheader("三、目标变量（电影票房）分布分析")
        # 原始票房分布直方图+箱线图
        fig_gross_dist = px.histogram(
            df,
            x="GROSS COLLECTION",
            title="原始电影票房整体分布（直方图 + 箱线图）",
            labels={"GROSS COLLECTION": "票房收入", "count": "影片数量"},
            color_discrete_sequence=['#4169E1'],
            marginal="box"  # 附带箱线图，查看分位数和异常值
        )
        fig_gross_dist.update_layout(bargap=0.1)
        st.plotly_chart(fig_gross_dist, use_container_width=True)

        # 对数变换后的票房分布（解决右偏问题，适配回归模型）
        df['log_gross'] = np.log1p(df['GROSS COLLECTION'])
        fig_log_gross_dist = px.histogram(
            df,
            x="log_gross",
            title="对数变换后电影票房分布（直方图 + 箱线图）",
            labels={"log_gross": "对数变换后票房", "count": "影片数量"},
            color_discrete_sequence=['#2E8B57'],
            marginal="box"
        )
        fig_log_gross_dist.update_layout(bargap=0.1)
        st.plotly_chart(fig_log_gross_dist, use_container_width=True)
        st.caption("💡 电影票房数据呈明显右偏分布，大部分影片票房集中在低区间，少数头部影片票房极高；对数变换后分布更接近正态分布，更适合用于回归模型训练")
        st.divider()

        # ===================== 新增：数值特征相关性热图 =====================
        st.subheader("四、数值特征相关性热图")
        # 筛选数值型特征
        numeric_cols = ['Year', 'runtime', 'RATING', 'metascore', 'votes', 'GROSS COLLECTION']
        numeric_df = df[numeric_cols].dropna()
        # 计算皮尔逊相关系数矩阵
        corr_matrix = numeric_df.corr()
        # 绘制交互式相关性热图
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title="数值特征相关性矩阵热图",
            labels=dict(x="特征", y="特征", color="相关系数")
        )
        fig_corr.update_layout(width=800, height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("💡 相关系数绝对值越接近1，特征间线性相关性越强；可以看到电影评价投票人数（votes）与大众评分（RATING）相关性最高，相关系数为 0.622，属于中等偏强正向相关，其次是评价投票人数（votes）与电影票房总收入（GROSS COLLECTION），相关系数 0.551")
        st.divider()

        # ===================== 原有图表（序号顺延） =====================
        # 5. 年份票房趋势 - 折线图
        st.subheader("5. 电影票房随年份变化趋势：呈现出长期增长、阶段性波动、头部效应加剧的趋势")
        year_gross = df.groupby('Year')['GROSS COLLECTION'].mean().dropna().reset_index()
        fig_line = px.line(
            year_gross,
            x="Year",
            y="GROSS COLLECTION",
            title="历年平均票房趋势",
            labels={"Year": "年份", "GROSS COLLECTION": "平均票房"}
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # 6. 评分与票房 - 散点图
        st.subheader("6. 电影评分与票房关系：评分与票房之间存在弱正相关关系，票房较高的电影普遍集中在 7.5-8.5 分区间")
        scatter_df = df[['RATING', 'GROSS COLLECTION']].dropna()
        fig_scatter = px.scatter(
            scatter_df,
            x="RATING",
            y="GROSS COLLECTION",
            title="评分 vs 票房",
            labels={"RATING": "电影评分", "GROSS COLLECTION": "票房收入"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # 7. 不同类型电影平均票房
        st.subheader("7. 不同类型电影的平均票房：冒险类影片平均票房最高")
        def split_genre(x):
            return str(x).split(',')
        df['genre_list'] = df['genre'].apply(split_genre)
        genre_gross = {}
        for idx, row in df.iterrows():
            gross = row['GROSS COLLECTION']
            for g in row['genre_list']:
                g = g.strip()
                if g and g != "nan":
                    if g not in genre_gross:
                        genre_gross[g] = []
                    genre_gross[g].append(gross)
        genre_avg = {k: np.mean(v) for k, v in genre_gross.items()}
        genre_df = pd.DataFrame(list(genre_avg.items()), columns=['类型', '平均票房'])
        fig_genre = px.bar(
            genre_df,
            x="类型",
            y="平均票房",
            title="各电影类型平均票房",
            color_discrete_sequence=['#4169E1']
        )
        st.plotly_chart(fig_genre, use_container_width=True)

        # 8. 导演Top10平均票房
        st.subheader("8. 导演Top10平均票房：Jon Watts导演作品平均票房最高")
        director_gross = df.groupby('DIRECTOR')['GROSS COLLECTION'].mean().sort_values(ascending=False).head(10).reset_index()
        fig_dir = px.bar(
            director_gross,
            x="DIRECTOR",
            y="GROSS COLLECTION",
            title="Top10 导演平均票房",
            labels={"DIRECTOR": "导演", "GROSS COLLECTION": "平均票房"},
            color_discrete_sequence=['#2E8B57']
        )
        st.plotly_chart(fig_dir, use_container_width=True)

        # 9. 电影时长 & 票房关系
        st.subheader("9. 电影时长 与 票房收入关系：100–180 分钟为票房黄金区间")
        runtime_gross_df = df[['runtime', 'GROSS COLLECTION']].dropna()
        fig_runtime = px.scatter(
            runtime_gross_df,
            x="runtime",
            y="GROSS COLLECTION",
            title="电影时长 vs 票房",
            labels={"runtime": "时长(分钟)", "GROSS COLLECTION": "票房收入"}
        )
        st.plotly_chart(fig_runtime, use_container_width=True)

        # 10. BBFC分级数量分布
        st.subheader("10. BBFC电影分级整体数量分布")
        mpaa_count = df['certificate'].value_counts().dropna().reset_index()
        mpaa_count.columns = ["分级", "数量"]
        fig_mpaa_cnt = px.bar(
            mpaa_count,
            x="分级",
            y="数量",
            title="各分级影片数量分布"
        )
        st.plotly_chart(fig_mpaa_cnt, use_container_width=True)

        # 11. BBFC分级 & 时长 箱线图
        st.subheader("11. BBFC分级 与 电影时长分布：大部分影片时长集中在110–135分钟")
        mpaa_runtime_df = df[['certificate', 'runtime']].dropna()
        fig_box1 = px.box(
            mpaa_runtime_df,
            x="certificate",
            y="runtime",
            title="分级与电影时长分布",
            labels={"certificate": "BBFC分级", "runtime": "时长(分钟)"}
        )
        st.plotly_chart(fig_box1, use_container_width=True)

        # 12. BBFC分级 & 票房 箱线图
        st.subheader("12. BBFC分级 与 电影票房收入分布：12A分级影片整体票房表现最优")
        mpaa_gross_df = df[['certificate', 'GROSS COLLECTION']].dropna()
        fig_box2 = px.box(
            mpaa_gross_df,
            x="certificate",
            y="GROSS COLLECTION",
            title="分级与票房收入分布",
            labels={"certificate": "BBFC分级", "GROSS COLLECTION": "票房收入"}
        )
        st.plotly_chart(fig_box2, use_container_width=True)

        st.info("💡 提示：所有图表数据均来自电影数据集，可根据需要拓展分析维度")

    # 票房预测页
    elif choice == "票房预测":
        st.title("电影票房预测模型")
        if predictor.model_scores and predictor.stacking_test_rmse is not None and predictor.stacking_test_r2 is not None:
            model_names_cn = [
                "线性回归",
                "梯度提升回归",
                "Ridge 回归",
                "Lasso回归",
                "随机森林回归",
                "Stacking模型融合"
            ]
            try:
                # 读取所有模型RMSE
                lr_rmse = predictor.model_scores["linear_regression"]["mean_cv_rmse"]
                gbr_rmse = predictor.model_scores["gradient_boosting"]["mean_cv_rmse"]
                ridge_rmse = predictor.model_scores["ridge_regression"]["mean_cv_rmse"]
                lasso_rmse = predictor.model_scores["lasso_regression"]["mean_cv_rmse"]
                rf_rmse = predictor.model_scores["random_forest"]["mean_cv_rmse"]
                stack_rmse = predictor.stacking_test_rmse

                rmse_data = [lr_rmse, gbr_rmse, ridge_rmse, lasso_rmse, rf_rmse, stack_rmse]

                # 读取所有模型R²
                lr_r2 = predictor.model_scores["linear_regression"]["mean_cv_r2"]
                gbr_r2 = predictor.model_scores["gradient_boosting"]["mean_cv_r2"]
                ridge_r2 = predictor.model_scores["ridge_regression"]["mean_cv_r2"]
                lasso_r2 = predictor.model_scores["lasso_regression"]["mean_cv_r2"]
                rf_r2 = predictor.model_scores["random_forest"]["mean_cv_r2"]
                stack_r2 = predictor.stacking_test_r2

                r2_data = [lr_r2, gbr_r2, ridge_r2, lasso_r2, rf_r2, stack_r2]

                # 左右两列并列展示RMSE、R²柱状图
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("各模型RMSE性能对比")
                    fig_rmse = px.bar(
                        x=model_names_cn,
                        y=rmse_data,
                        color_discrete_sequence=['#642EFE'],
                        title="机器学习电影票房预测-RMSE对比",
                        labels={"x": "模型", "y": "RMSE(均方根误差)"}
                    )
                    fig_rmse.update_traces(texttemplate='%{y:.4f}', textposition='outside')
                    fig_rmse.update_layout(height=400)
                    st.plotly_chart(fig_rmse, use_container_width=True)
                    st.info("RMSE越小，模型预测误差越小，预测精度越高")

                with col2:
                    st.subheader("各模型R²拟合优度对比")
                    fig_r2 = px.bar(
                        x=model_names_cn,
                        y=r2_data,
                        color_discrete_sequence=['#2E8B57'],
                        title="机器学习电影票房预测-R²决定系数对比",
                        labels={"x": "模型", "y": "R²(决定系数)"}
                    )
                    fig_r2.update_traces(texttemplate='%{y:.4f}', textposition='outside')
                    fig_r2.update_layout(height=400)
                    st.plotly_chart(fig_r2, use_container_width=True)
                    st.info("R²越接近1，模型对数据的解释能力越强，拟合效果越好")

            except KeyError:
                st.warning("模型性能指标不完整，请删除所有.pkl文件后重新完整训练模型！")
        else:
            st.warning("暂无训练完成的模型性能数据，请先运行模型训练流程生成模型文件！")
        st.divider()

        # 缓存读取下拉选项，不用每次读取csv去重
        df_raw = load_raw_csv()
        directors, actors1, actors2 = get_select_options(df_raw)

        model_map = {
            "梯度提升回归模型": "gradient_boosting",
            "Stacking融合模型": "stacking_model"
            "随机森林回归模型": "random_forest",
            "线性回归模型": "linear_regression",
            "Ridge回归模型": "ridge_regression",
            "Lasso回归模型": "lasso_regression",
            
            
        }
        select_cn = st.selectbox("请选择预测模型", list(model_map.keys()))
        select_model = model_map[select_cn]

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                # 预设肖申克的救赎：1994年
                year = st.number_input("上映年份", min_value=1900, max_value=2100, value=1994)
                # 预设时长142分钟
                runtime = st.number_input("电影时长(分钟)", value=142)
                # 预设评分9.3
                rating = st.number_input("电影评分", min_value=0.0, max_value=10.0, value=9.3, step=0.1)
            with col2:
                # 预设影评分数81
                metascore = st.number_input("影评分数", min_value=0, max_value=100, value=81)
                # 预设投票数2603814
                votes = st.number_input("投票数", value=2603814)
                # 从分级专属编码器读取所有BBFC分级
                if predictor.le is not None:
                    cert_options = sorted(predictor.le.classes_.tolist())
                else:
                    cert_options = ["unknown"]
                # 默认选中分级15
                cert_index = cert_options.index("15") if "15" in cert_options else 0
                certificate = st.selectbox("BBFC电影分级", cert_options, index=cert_index)
            with col3:
                # 默认导演 Frank Darabont
                dir_index = directors.index("Frank Darabont") if "Frank Darabont" in directors else 0
                director = st.selectbox("导演", directors, index=dir_index)
                # 默认主演1 Tim Robbins
                act1_index = actors1.index("Tim Robbins") if "Tim Robbins" in actors1 else 0
                actor1 = st.selectbox("主演1", actors1, index=act1_index)
                # 默认主演2 Morgan Freeman
                act2_index = actors2.index("Morgan Freeman") if "Morgan Freeman" in actors2 else 0
                actor2 = st.selectbox("主演2", actors2, index=act2_index)

            # 默认选中剧情类Drama
            genre = st.multiselect(
                "电影类型",
                ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"],
                default=["Drama"]
            )
            submit = st.form_submit_button("开始预测")

        if submit:
            # 加载动画优化交互感知
            with st.spinner("模型正在计算票房，请稍候..."):
                try:
                    data = {
                        "Year": [year],
                        "runtime": [runtime],
                        "certificate": [certificate],
                        "RATING": [rating],
                        "metascore": [metascore],
                        "votes": [votes],
                        "DIRECTOR": [director],
                        "ACTOR 1": [actor1],
                        "ACTOR 2": [actor2]
                    }
                    if not genre:
                        genre = ["unknown"]
                    data["genre"] = [genre]
                    input_df = pd.DataFrame(data)

                    if predictor.le:
                        if certificate not in predictor.le.classes_:
                            default_label = "unknown" if "unknown" in predictor.le.classes_ else predictor.le.classes_[0]
                            input_df['certificate'] = predictor.le.transform([default_label])[0]
                        else:
                            input_df['certificate'] = predictor.le.transform(input_df['certificate'].astype(str))

                        if director not in predictor.le.classes_:
                            default_dir = "未知导演" if "未知导演" in predictor.le.classes_ else predictor.le.classes_[0]
                            input_df['DIRECTOR'] = predictor.le.transform([default_dir])[0]
                        else:
                            input_df['DIRECTOR'] = predictor.le.transform(input_df['DIRECTOR'].astype(str))

                        if actor1 not in predictor.le.classes_:
                            default_act = "未知演员" if "未知演员" in predictor.le.classes_ else predictor.le.classes_[0]
                            input_df['ACTOR 1'] = predictor.le.transform([default_act])[0]
                        else:
                            input_df['ACTOR 1'] = predictor.le.transform(input_df['ACTOR 1'].astype(str))

                        if actor2 not in predictor.le.classes_:
                            default_act2 = "未知演员" if "未知演员" in predictor.le.classes_ else predictor.le.classes_[0]
                            input_df['ACTOR 2'] = predictor.le.transform([default_act2])[0]
                        else:
                            input_df['ACTOR 2'] = predictor.le.transform(input_df['ACTOR 2'].astype(str))
                    else:
                        input_df['certificate'] = 0
                        input_df['DIRECTOR'] = 0
                        input_df['ACTOR 1'] = 0
                        input_df['ACTOR 2'] = 0

                    if predictor.mlb:
                        genre_arr = predictor.mlb.transform(input_df['genre'])
                        genre_df = pd.DataFrame(genre_arr, columns=predictor.mlb.classes_)
                        input_df = pd.concat([input_df.drop('genre', axis=1), genre_df], axis=1)
                    else:
                        for col in predictor.feature_columns:
                            if col not in input_df.columns:
                                input_df[col] = 0

                    if predictor.feature_columns:
                        input_df = input_df[predictor.feature_columns]
                    input_df = input_df.astype(float)

                    pred_log = predictor.predict_by_model(input_df.values, select_model)
                    pred_real = np.expm1(pred_log[0])
                    # 转换为百万美元，保留两位小数展示
                    pred_million = pred_real / 1_000_000
                    st.success(f"当前使用模型：{select_cn}")
                    st.markdown(f"### 预测票房：:red[${pred_million:.2f}M]（折合 {pred_real:,.0f} 美元）")
                except Exception as e:
                    st.error(f"预测异常：{str(e)}，启用模拟结果")
                    pred = 371300633
                    pred_million = pred / 1_000_000
                    st.markdown(f"### 模拟预测票房：:red[${pred_million:.2f}M]（折合 {pred:,.0f} 美元）")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    # 所有模型文件清单
    all_model_files = [
        "linear_regression.pkl",
        "ridge_regression.pkl",
        "lasso_regression.pkl",
        "random_forest.pkl",
        "gradient_boosting.pkl",
        "stacking_model.pkl",
        "model_scores.pkl",
        "stacking_rmse.pkl",
        "stacking_r2.pkl",
        "feature_columns.pkl",
        "label_encoder.pkl",
        "multi_label_binarizer.pkl"
    ]

    # 判断是否全部模型文件存在
    all_model_exist = all([os.path.exists(f) for f in all_model_files])
    CSV_FILE = "电影数据.csv"

    # 缺失模型文件则重新训练
    if not all_model_exist and os.path.exists(CSV_FILE):
        print("⚠️ 缺失模型缓存文件，开始训练模型，请稍等...")
        X, y, le_cert, mlb = load_and_preprocess(CSV_FILE)
        temp_predictor = MovieBoxOfficePredictor()
        temp_predictor.le = le_cert
        temp_predictor.mlb = mlb
        temp_predictor.train_with_kfold(X, y)
        temp_predictor.train_stacking(X, y)
        temp_predictor.save_models()
        print("✅ 模型训练、编码器保存完成！")
    elif all_model_exist:
        print("✅ 检测到完整缓存模型文件，跳过训练，直接加载复用")
    else:
        print("❌ 未找到 电影数据.csv，进入模拟预测模式")

    # 全局加载模型
    predictor = get_predictor_instance()
    print("🚀 启动电影票房系统...")
    run_streamlit_app(predictor)
