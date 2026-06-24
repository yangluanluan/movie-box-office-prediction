# 依赖安装（首次运行执行）
# pip install streamlit pandas numpy scikit-learn joblib plotly scipy
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
from scipy.stats.mstats import winsorize
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# -------------------------- 全局缓存工具 --------------------------
@st.cache_data
def load_raw_csv(csv_path="电影数据.csv"):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data
def get_select_options(df):
    directors = sorted(df['DIRECTOR'].dropna().unique().tolist())
    actors1 = sorted(df['ACTOR 1'].dropna().unique().tolist())
    actors2 = sorted(df['ACTOR 2'].dropna().unique().tolist())
    return directors, actors1, actors2

@st.cache_resource
def get_predictor_instance():
    pred = MovieBoxOfficePredictor()
    pred.load_models()
    return pred

# ===================== 模型预测类（修复维度不匹配BUG） =====================
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
        self.le_certificate = None
        self.le_director = None
        self.le_actor1 = None
        self.le_actor2 = None
        self.mlb = None
        self.stacking_test_rmse = None
        self.stacking_test_r2 = None
        self.raw_y = None

    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def train_with_kfold(self, X, y_log, y_raw, n_splits=5, random_state=42):
        self.feature_columns = list(X.columns)
        X_arr = X.values
        y_log_arr = y_log.values
        self.raw_y = y_raw.values
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        results = {}
        oof_list = []

        for model_name, model in self.models.items():
            cv_rmse_scores = []
            cv_r2_scores = []
            oof_train = np.zeros(len(X_arr))
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_arr)):
                X_train = X_arr[train_idx]
                X_val = X_arr[val_idx]
                y_train_log = y_log_arr[train_idx]
                y_val_log = y_log_arr[val_idx]
                y_val_real = self.raw_y[val_idx]

                model_fold = model.__class__(**model.get_params())
                model_fit = model_fold.fit(X_train, y_train_log)
                val_pred_log = model_fit.predict(X_val)
                # 修复1：压缩为一维数组，解决维度不匹配赋值报错
                val_pred_log = val_pred_log.ravel()
                val_pred_real = np.expm1(val_pred_log)
                oof_train[val_idx] = val_pred_log

                fold_rmse = self.rmse(y_val_real, val_pred_real)
                fold_r2 = r2_score(y_val_real, val_pred_real)
                cv_rmse_scores.append(fold_rmse)
                cv_r2_scores.append(fold_r2)

            results[model_name] = {
                'cv_rmse_scores': cv_rmse_scores,
                'cv_r2_scores': cv_r2_scores,
                'oof_train': oof_train,
                'mean_cv_rmse': np.mean(cv_rmse_scores),
                'std_cv_rmse': np.std(cv_rmse_scores),
                'mean_cv_r2': np.mean(cv_r2_scores)
            }
            oof_list.append(oof_train)
            full_model = model.__class__(**model.get_params())
            full_model.fit(X_arr, y_log_arr)
            self.trained_models[model_name] = full_model

        self.full_oof = np.column_stack(oof_list)
        self.model_scores = results
        return results

    def train_stacking(self, X, y_log, y_raw, test_size=0.2):
        if self.full_oof is None:
            raise Exception("请先训练基模型！")
        y_log_arr = y_log.values
        y_raw_arr = y_raw.values
        X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
            self.full_oof, y_log_arr, y_raw_arr, test_size=test_size, random_state=42
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X_train):
            X_fold = X_train[train_idx]
            y_fold = y_train_log[train_idx]
            stacking_fold = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=32, min_samples_split=2)
            stacking_fold.fit(X_fold, y_fold)

        self.stacking_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=32, min_samples_split=2)
        self.stacking_model.fit(X_train, y_train_log)

        stacking_pred_log = self.stacking_model.predict(X_test)
        stacking_pred_real = np.expm1(stacking_pred_log)
        self.stacking_test_rmse = self.rmse(y_test_raw, stacking_pred_real)
        self.stacking_test_r2 = r2_score(y_test_raw, stacking_pred_real)
        print(f"\n=== Stacking 堆叠模型训练完成（原始票房尺度）===")
        print(f"Stacking RMSE(美元): {self.stacking_test_rmse:.2f}")
        print(f"Stacking R²: {self.stacking_test_r2:.4f}")

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
        if self.le_certificate:
            joblib.dump(self.le_certificate, "le_certificate.pkl")
        if self.le_director:
            joblib.dump(self.le_director, "le_director.pkl")
        if self.le_actor1:
            joblib.dump(self.le_actor1, "le_actor1.pkl")
        if self.le_actor2:
            joblib.dump(self.le_actor2, "le_actor2.pkl")
        if self.mlb:
            joblib.dump(self.mlb, "multi_label_binarizer.pkl")
        joblib.dump(self.stacking_test_rmse, "stacking_rmse.pkl")
        joblib.dump(self.stacking_test_r2, "stacking_r2.pkl")
        joblib.dump(self.model_scores, "model_scores.pkl")
        print("\n✅ 模型、原始尺度评估指标、所有分类编码器已保存")

    def load_models(self):
        self.trained_models = {}
        for model_name in self.models.keys():
            fp = f"{model_name}.pkl"
            if os.path.exists(fp):
                self.trained_models[model_name] = joblib.load(fp)
        if os.path.exists("stacking_model.pkl"):
            self.stacking_model = joblib.load("stacking_model.pkl")
        if os.path.exists("stacking_rmse.pkl"):
            self.stacking_test_rmse = joblib.load("stacking_rmse.pkl")
        if os.path.exists("stacking_r2.pkl"):
            self.stacking_test_r2 = joblib.load("stacking_r2.pkl")
        if os.path.exists("model_scores.pkl"):
            self.model_scores = joblib.load("model_scores.pkl")
        if os.path.exists("feature_columns.pkl"):
            self.feature_columns = joblib.load("feature_columns.pkl")
        if os.path.exists("le_certificate.pkl"):
            self.le_certificate = joblib.load("le_certificate.pkl")
        if os.path.exists("le_director.pkl"):
            self.le_director = joblib.load("le_director.pkl")
        if os.path.exists("le_actor1.pkl"):
            self.le_actor1 = joblib.load("le_actor1.pkl")
        if os.path.exists("le_actor2.pkl"):
            self.le_actor2 = joblib.load("le_actor2.pkl")
        if os.path.exists("multi_label_binarizer.pkl"):
            self.mlb = joblib.load("multi_label_binarizer.pkl")
        print("✅ 模型与所有分类编码器加载完成")

# ===================== 工具函数 =====================
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

# 稀有类别合并：降低导演、演员高基数特征权重
def freq_encode_rare_cat(series, min_count=3):
    freq = series.value_counts()
    rare_cats = freq[freq < min_count].index
    return series.replace(rare_cats, "其他_稀有类别")

# ===================== 数据预处理（异常值缩尾+特征降噪） =====================
def load_and_preprocess(csv_path):
    df = load_raw_csv(csv_path)
    df['GROSS COLLECTION'] = df['GROSS COLLECTION'].apply(parse_gross)
    df = df.dropna(subset=['GROSS COLLECTION']).reset_index(drop=True)

    # 票房上下1% Winsor缩尾，剔除极端爆款异常值
    df['GROSS_WINSOR'] = winsorize(df['GROSS COLLECTION'], limits=[0.01, 0.01])
    y_raw = df['GROSS_WINSOR'].copy()
    y_log = np.log1p(y_raw)

    feature_cols = [
        'Year', 'runtime', 'certificate', 'genre', 'RATING', 'metascore', 'votes',
        'DIRECTOR', 'ACTOR 1', 'ACTOR 2'
    ]
    X = df[feature_cols].copy()

    X['Year'] = X['Year'].apply(extract_year)
    X['runtime'] = X['runtime'].astype(str).str.replace(' min', '').astype(float)
    X['votes'] = X['votes'].astype(str).str.replace(',', '').astype(float)

    X['DIRECTOR'] = X['DIRECTOR'].fillna("未知导演")
    X['ACTOR 1'] = X['ACTOR 1'].fillna("未知演员")
    X['ACTOR 2'] = X['ACTOR 2'].fillna("未知演员")

    drop_cols = ['Year', 'runtime', 'RATING', 'metascore', 'votes']
    combined = pd.concat([X, y_log, y_raw], axis=1)
    combined = combined.dropna(subset=drop_cols).reset_index(drop=True)
    X = combined[feature_cols].copy()
    y_log = combined['GROSS_WINSOR'].apply(np.log1p)
    y_raw = combined['GROSS_WINSOR']

    X['certificate'] = X['certificate'].fillna("unknown")

    # 稀有导演、演员合并降噪
    X['DIRECTOR'] = freq_encode_rare_cat(X['DIRECTOR'], min_count=3)
    X['ACTOR 1'] = freq_encode_rare_cat(X['ACTOR 1'], min_count=3)
    X['ACTOR 2'] = freq_encode_rare_cat(X['ACTOR 2'], min_count=3)

    # 独立编码器
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
    print(f"缩尾清洗后有效样本量：{X.shape[0]}")
    return X, y_log, y_raw, le_certificate, le_director, le_actor1, le_actor2, mlb

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

    if choice == "首页":
        st.title("基于Python的电影票房分析预测系统")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("系统介绍")
            st.write("""
            票房作为衡量电影能否盈利的重要指标，受诸多因素共同作用影响，
            且其影响机制较为复杂，票房的准确预测难度较大。本项目首先将影响电影类型、上映档期、导演、演员等因素量化处理并进行可视化分析。
            数据预处理阶段对极端高票房样本做Winsor缩尾处理，对导演、演员稀有类别合并降噪，缓解爆款特征过拟合、普通影片票房高估问题；
            模型训练基于原始票房美元尺度计算RMSE、R²评估指标，预测结果经过对数逆变换还原真实票房，具备业务可解释性。
            """)
        with col2:
            st.info("本系统已取消登录验证，可直接前往【票房预测】或【票房分析】使用功能")

    elif choice == "票房分析":
        st.title("📊 电影票房数据分析")
        st.write("以下是电影数据的可视化分析图表，帮助你理解票房的影响因素")
        st.divider()
        df_raw = load_raw_csv()
        st.subheader("一、原始数据集预览（前10行）")
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.caption(f"数据集总记录数：{len(df_raw)} 条")
        st.divider()

        field_info = [
            {"字段名": "Year", "中文名称": "上映年份", "字段含义": "电影正式上映的年份"},
            {"字段名": "runtime", "中文名称": "影片时长", "字段含义": "电影正片时长，单位：分钟"},
            {"字段名": "certificate", "中文名称": "影片分级", "字段含义": "英国BBFC电影分级，用于划分观影人群"},
            {"字段名": "genre", "中文名称": "电影类型", "字段含义": "影片题材类型，支持多类型组合"},
            {"字段名": "RATING", "中文名称": "大众评分", "字段含义": "普通观众综合评分，满分10分"},
            {"字段名": "metascore", "中文名称": "影评分数", "字段含义": "专业影评人打分，满分100分"},
            {"字段名": "votes", "中文名称": "评价人数", "字段含义": "参与评分用户总数"},
            {"字段名": "DIRECTOR", "中文名称": "导演", "字段含义": "电影主创导演"},
            {"字段name": "ACTOR 1", "中文名称": "主演1", "字段含义": "第一主演"},
            {"字段名": "ACTOR 2", "中文名称": "主演2", "字段含义": "第二主演"},
            {"字段名": "GROSS COLLECTION", "中文名称": "总票房", "字段含义": "全球总票房（美元）"}
        ]
        field_df = pd.DataFrame(field_info)
        st.dataframe(field_df, use_container_width=True, hide_index=True)
        st.divider()

        df = df_raw.copy()
        df['GROSS COLLECTION'] = df['GROSS COLLECTION'].apply(parse_gross)
        df = df.dropna(subset=['GROSS COLLECTION']).reset_index(drop=True)
        df['Year'] = df['Year'].apply(extract_year)
        df['runtime'] = df['runtime'].astype(str).str.replace(' min', '').astype(float)
        df['votes'] = df['votes'].astype(str).str.replace(',', '').astype(float)

        st.subheader("三、票房分布（缩尾前原始数据）")
        fig_gross_dist = px.histogram(df, x="GROSS COLLECTION", marginal="box",
                                      title="原始票房分布（存在极端高票房异常值）",
                                      labels={"GROSS COLLECTION": "票房(美元)"})
        st.plotly_chart(fig_gross_dist, use_container_width=True)
        df['log_gross'] = np.log1p(df['GROSS COLLECTION'])
        fig_log = px.histogram(df, x="log_gross", marginal="box", title="对数变换后票房分布")
        st.plotly_chart(fig_log, use_container_width=True)
        st.caption("💡 优化方案：先对票房做1%上下Winsor缩尾剔除极端异常值，再进行对数变换，避免头部爆款样本主导模型训练")
        st.divider()

        numeric_cols = ['Year', 'runtime', 'RATING', 'metascore', 'votes', 'GROSS COLLECTION']
        numeric_df = df[numeric_cols].dropna()
        corr_matrix = numeric_df.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                             title="数值特征相关性热图")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.divider()

        year_gross = df.groupby('Year')['GROSS COLLECTION'].mean().dropna().reset_index()
        fig_line = px.line(year_gross, x="Year", y="GROSS COLLECTION", title="历年平均票房趋势")
        st.plotly_chart(fig_line, use_container_width=True)

        fig_scatter = px.scatter(df, x="RATING", y="GROSS COLLECTION", title="评分与票房散点图")
        st.plotly_chart(fig_scatter, use_container_width=True)

        def split_genre(x):
            return str(x).split(',')
        df['genre_list'] = df['genre'].apply(split_genre)
        genre_gross = {}
        for _, row in df.iterrows():
            g = row['GROSS COLLECTION']
            for t in row['genre_list']:
                t = t.strip()
                if t and t != "nan":
                    genre_gross[t] = genre_gross.get(t, []) + [g]
        genre_avg = {k: np.mean(v) for k, v in genre_gross.items()}
        genre_df = pd.DataFrame(list(genre_avg.items()), columns=['类型', '平均票房'])
        fig_genre = px.bar(genre_df, x="类型", y="平均票房", title="各类型电影平均票房")
        st.plotly_chart(fig_genre, use_container_width=True)

        director_gross = df.groupby('DIRECTOR')['GROSS COLLECTION'].mean().sort_values(ascending=False).head(10).reset_index()
        fig_dir = px.bar(director_gross, x="DIRECTOR", y="GROSS COLLECTION", title="TOP10导演平均票房")
        st.plotly_chart(fig_dir, use_container_width=True)

        fig_runtime = px.scatter(df, x="runtime", y="GROSS COLLECTION", title="影片时长与票房")
        st.plotly_chart(fig_runtime, use_container_width=True)

        cert_cnt = df['certificate'].value_counts().reset_index()
        cert_cnt.columns = ["分级", "数量"]
        fig_cert = px.bar(cert_cnt, x="分级", y="数量", title="各分级影片数量")
        st.plotly_chart(fig_cert, use_container_width=True)

        fig_box_cert_gross = px.box(df, x="certificate", y="GROSS COLLECTION", title="分级票房分布箱线图")
        st.plotly_chart(fig_box_cert_gross, use_container_width=True)

    elif choice == "票房预测":
        st.title("电影票房预测模型（基于原始票房尺度评估）")
        if predictor.model_scores and predictor.stacking_test_rmse is not None:
            model_names_cn = ["线性回归", "Ridge回归", "Lasso回归", "随机森林回归", "梯度提升回归", "Stacking融合模型"]
            try:
                lr_rmse = predictor.model_scores["linear_regression"]["mean_cv_rmse"]
                ridge_rmse = predictor.model_scores["ridge_regression"]["mean_cv_rmse"]
                lasso_rmse = predictor.model_scores["lasso_regression"]["mean_cv_rmse"]
                rf_rmse = predictor.model_scores["random_forest"]["mean_cv_rmse"]
                gbr_rmse = predictor.model_scores["gradient_boosting"]["mean_cv_rmse"]
                stack_rmse = predictor.stacking_test_rmse

                lr_r2 = predictor.model_scores["linear_regression"]["mean_cv_r2"]
                ridge_r2 = predictor.model_scores["ridge_regression"]["mean_cv_r2"]
                lasso_r2 = predictor.model_scores["lasso_regression"]["mean_cv_r2"]
                rf_r2 = predictor.model_scores["random_forest"]["mean_cv_r2"]
                gbr_r2 = predictor.model_scores["gradient_boosting"]["mean_cv_r2"]
                stack_r2 = predictor.stacking_test_r2

                rmse_data = [lr_rmse, ridge_rmse, lasso_rmse, rf_rmse, gbr_rmse, stack_rmse]
                r2_data = [lr_r2, ridge_r2, lasso_r2, rf_r2, gbr_r2, stack_r2]

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("各模型RMSE对比（原始票房·美元）")
                    fig_rmse = px.bar(x=model_names_cn, y=rmse_data, title="模型均方根误差",
                                      labels={"y": "RMSE(美元)"})
                    fig_rmse.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                    st.plotly_chart(fig_rmse, use_container_width=True)
                    st.info("基于真实票房计算，可直观衡量预测与实际票房的美元误差")
                with col2:
                    st.subheader("各模型R²拟合优度对比")
                    fig_r2 = px.bar(x=model_names_cn, y=r2_data, title="模型决定系数R²")
                    fig_r2.update_traces(texttemplate='%{y:.4f}', textposition='outside')
                    st.plotly_chart(fig_r2, use_container_width=True)
            except KeyError:
                st.warning("指标异常，请删除所有.pkl文件后重新训练！")
        else:
            st.warning("暂无训练完成的模型指标")
        st.divider()

        df_raw = load_raw_csv()
        directors, actors1, actors2 = get_select_options(df_raw)
        model_map = {
            "线性回归模型": "linear_regression",
            "Ridge回归模型": "ridge_regression",
            "Lasso回归模型": "lasso_regression",
            "随机森林回归模型": "random_forest",
            "梯度提升回归模型": "gradient_boosting",
            "Stacking融合模型": "stacking_model"
        }
        select_cn = st.selectbox("选择预测模型", list(model_map.keys()))
        select_model = model_map[select_cn]

        with st.form("pred_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                year = st.number_input("上映年份", min_value=1900, max_value=2100, value=1994)
                runtime = st.number_input("影片时长(分钟)", value=142)
                rating = st.number_input("大众评分", min_value=0.0, max_value=10.0, value=9.3, step=0.1)
            with col2:
                metascore = st.number_input("影评分数", min_value=0, max_value=100, value=81)
                votes = st.number_input("评价人数", value=2603814)
                cert_options = sorted(predictor.le_certificate.classes_.tolist()) if predictor.le_certificate else ["unknown"]
                cert_idx = cert_options.index("15") if "15" in cert_options else 0
                certificate = st.selectbox("BBFC分级", cert_options, index=cert_idx)
            with col3:
                dir_idx = directors.index("Frank Darabont") if "Frank Darabont" in directors else 0
                director = st.selectbox("导演", directors, index=dir_idx)
                act1_idx = actors1.index("Tim Robbins") if "Tim Robbins" in actors1 else 0
                actor1 = st.selectbox("主演1", actors1, index=act1_idx)
                act2_idx = actors2.index("Morgan Freeman") if "Morgan Freeman" in actors2 else 0
                actor2 = st.selectbox("主演2", actors2, index=act2_idx)

            genre = st.multiselect("电影类型",
                                   ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"],
                                   default=["Drama"])
            submit = st.form_submit_button("开始预测")

        if submit:
            with st.spinner("预测计算中..."):
                try:
                    data = {
                        "Year": [year], "runtime": [runtime], "certificate": [certificate],
                        "RATING": [rating], "metascore": [metascore], "votes": [votes],
                        "DIRECTOR": [director], "ACTOR 1": [actor1], "ACTOR 2": [actor2], "genre": [genre if genre else ["unknown"]]
                    }
                    input_df = pd.DataFrame(data)
                    input_df['certificate'] = predictor.le_certificate.transform([certificate])[0]
                    rare_label = "其他_稀有类别"

                    # 修复2：传入正确的标签编码器，不再传入模型特征权重
                    def safe_encode(col_val, encoder):
                        if col_val not in encoder.classes_:
                            return encoder.transform([rare_label if rare_label in encoder.classes_ else encoder.classes_[0]])[0]
                        return encoder.transform([col_val])[0]

                    input_df['DIRECTOR'] = safe_encode(director, predictor.le_director)
                    input_df['ACTOR 1'] = safe_encode(actor1, predictor.le_actor1)
                    input_df['ACTOR 2'] = safe_encode(actor2, predictor.le_actor2)

                    genre_arr = predictor.mlb.transform(input_df['genre'])
                    genre_df = pd.DataFrame(genre_arr, columns=predictor.mlb.classes_)
                    input_df = pd.concat([input_df.drop('genre', axis=1), genre_df], axis=1)
                    for col in predictor.feature_columns:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    input_df = input_df[predictor.feature_columns].astype(float)

                    pred_log = predictor.predict_by_model(input_df.values, select_model)
                    pred_real = np.expm1(pred_log[0])
                    pred_million = pred_real / 1_000_000
                    st.success(f"使用模型：{select_cn}")
                    st.markdown(f"### 预测票房：:red[${pred_million:.2f}M]（{pred_real:,.0f} 美元）")
                except Exception as e:
                    st.error(f"预测异常：{str(e)}")
                    pred = 371300633
                    st.markdown(f"### 模拟预测票房：${pred/1_000_000:.2f}M")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    # 清理所有旧缓存文件
    all_model_files = [
        "linear_regression.pkl", "ridge_regression.pkl", "lasso_regression.pkl",
        "random_forest.pkl", "gradient_boosting.pkl", "stacking_model.pkl",
        "model_scores.pkl", "stacking_rmse.pkl", "stacking_r2.pkl",
        "feature_columns.pkl", "le_certificate.pkl", "le_director.pkl",
        "le_actor1.pkl", "le_actor2.pkl", "multi_label_binarizer.pkl",
        "label_encoder.pkl"
    ]
    for file in all_model_files:
        if os.path.exists(file):
            os.remove(file)
    print("✅ 已清理所有旧版模型缓存文件")

    csv_file = "电影数据.csv"
    if os.path.exists(csv_file):
        print("开始执行数据清洗+模型训练...")
        X, y_log, y_raw, le_cert, le_dir, le_act1, le_act2, mlb = load_and_preprocess(csv_file)
        pred = MovieBoxOfficePredictor()
        pred.le_certificate = le_cert
        pred.le_director = le_dir
        pred.le_actor1 = le_act1
        pred.le_actor2 = le_act2
        pred.mlb = mlb
        pred.train_with_kfold(X, y_log, y_raw)
        pred.train_stacking(X, y_log, y_raw)
        pred.save_models()
        print("✅ 模型训练保存完成")

    predictor = get_predictor_instance()
    run_streamlit_app(predictor)
