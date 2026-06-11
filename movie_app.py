# 依赖安装（首次运行执行）
# pip install streamlit pandas numpy scikit-learn joblib

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

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
            cv_scores = []
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
                fold_rmse = self.rmse(y_val, val_pred)
                cv_scores.append(fold_rmse)

            results[model_name] = {
                'cv_scores': cv_scores,
                'oof_train': oof_train,
                'mean_cv_rmse': np.mean(cv_scores),
                'std_cv_rmse': np.std(cv_scores)
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
        print("\n✅ 所有模型和特征配置已保存")

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
        if os.path.exists("feature_columns.pkl"):
            self.feature_columns = joblib.load("feature_columns.pkl")
        if os.path.exists("label_encoder.pkl"):
            self.le = joblib.load("label_encoder.pkl")
        if os.path.exists("multi_label_binarizer.pkl"):
            self.mlb = joblib.load("multi_label_binarizer.pkl")
        print("✅ 所有基模型和特征配置加载完成")

# ===================== 数据预处理函数 =====================
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
    df = pd.read_csv(csv_path)
    # 清除所有列名首尾空格
    df.columns = [c.strip() for c in df.columns]
    
    df['GROSS COLLECTION'] = df['GROSS COLLECTION'].apply(parse_gross)
    df = df.dropna(subset=['GROSS COLLECTION']).reset_index(drop=True)

    feature_cols = [
        'Year', 'runtime', 'certificate', 'genre', 'RATING', 'metascore', 'votes',
        'DIRECTOR', 'ACTOR 1', 'ACTOR 2'
    ]
    X = df[feature_cols].copy()
    y_raw = df['GROSS COLLECTION']
    y = np.log1p(y_raw)

    # 基础清洗
    X['Year'] = X['Year'].apply(extract_year)
    X['runtime'] = X.astype(str)['runtime'].str.replace(' min', '').astype(float)
    X['votes'] = X.astype(str)['votes'].str.replace(',', '').astype(float)

    # 填充缺失值
    X['DIRECTOR'] = X['DIRECTOR'].fillna("未知导演")
    X['ACTOR 1'] = X['ACTOR 1'].fillna("未知演员")
    X['ACTOR 2'] = X['ACTOR 2'].fillna("未知演员")

    drop_cols = ['Year', 'runtime', 'RATING', 'metascore', 'votes']
    combined = pd.concat([X, y], axis=1)
    combined = combined.dropna(subset=drop_cols).reset_index(drop=True)
    X = combined[feature_cols].copy()
    y = combined['GROSS COLLECTION']

    # 分类特征编码
    X['certificate'] = X['certificate'].fillna("unknown")
    le = LabelEncoder()
    X['certificate'] = le.fit_transform(X['certificate'].astype(str))
    X['DIRECTOR'] = le.fit_transform(X['DIRECTOR'].astype(str))
    X['ACTOR 1'] = le.fit_transform(X['ACTOR 1'].astype(str))
    X['ACTOR 2'] = le.fit_transform(X['ACTOR 2'].astype(str))

    # 电影类型多标签编码
    def split_genre(x):
        return str(x).split(',')
    X['genre'] = X['genre'].apply(split_genre)
    mlb = MultiLabelBinarizer()
    genre_arr = mlb.fit_transform(X['genre'])
    genre_df = pd.DataFrame(genre_arr, columns=mlb.classes_)

    X = pd.concat([X.drop('genre', axis=1), genre_df], axis=1)
    X = X.astype(float)
    return X, y, le, mlb

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
    menu = ["首页", "票房预测"]
    choice = st.sidebar.selectbox("导航", menu)

    # 首页
    if choice == "首页":
        st.title("基于Python的电影票房分析预测系统")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("系统介绍")
            st.write("""
            票房作为衡量电影能否盈利的重要指标，受诸多因素共同作用影响，
            且其影响机制较为复杂，票房的准确预测难度较大。本项目利用电影数据集构建票房预测模型，
            采用多种回归模型 + Stacking堆叠集成学习实现票房预测。
            """)
            st.image(
                "https://cdn.pixabay.com/photo/2016/03/31/18/36/cinema-1294969_1280.jpg",
                caption="全球电影票房排行榜", width="stretch"
            )
        with col2:
            st.info("本系统已取消登录验证，可直接前往【票房预测】使用功能")

    # 票房预测页
    elif choice == "票房预测":
        st.title("电影票房预测模型")

        # 读取CSV并清除列名空格
        df = pd.read_csv("电影数据.csv")
        df.columns = [c.strip() for c in df.columns]

        directors = sorted(df['DIRECTOR'].dropna().unique().tolist())
        actors1 = sorted(df['ACTOR 1'].dropna().unique().tolist())
        actors2 = sorted(df['ACTOR 2'].dropna().unique().tolist())

        model_list = [
            "linear_regression",
            "ridge_regression",
            "lasso_regression",
            "random_forest",
            "gradient_boosting",
            "stacking_model"
        ]
        select_model = st.selectbox("请选择预测模型（含Stacking堆叠模型）", model_list)

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                year = st.number_input("上映年份", min_value=1900, max_value=2100, value=2015)
                runtime = st.number_input("电影时长(分钟)", value=120)
                rating = st.number_input("电影评分", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
            with col2:
                metascore = st.number_input("影评分数", min_value=0, max_value=100, value=70)
                votes = st.number_input("投票数", value=100000)
                certificate = st.selectbox("分级", ["G", "PG", "PG-13", "R", "unknown"])
            with col3:
                director = st.selectbox("导演", directors)
                actor1 = st.selectbox("主演1", actors1)
                actor2 = st.selectbox("主演2", actors2)

            genre = st.multiselect("电影类型", ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"])
            submit = st.form_submit_button("预测票房")

        if submit:
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

                # 编码转换
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
                st.success(f"当前使用模型：{select_model}")
                st.markdown(f"### 预测票房：:red[{pred_real:.0f}] 元")
            except Exception as e:
                st.error(f"预测异常：{str(e)}，启用模拟结果")
                pred = 371300633
                st.markdown(f"### 模拟预测票房：:red[{pred}] 万元")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    predictor = MovieBoxOfficePredictor()

    all_model_files = [
        "linear_regression.pkl",
        "ridge_regression.pkl",
        "lasso_regression.pkl",
        "random_forest.pkl",
        "gradient_boosting.pkl",
        "stacking_model.pkl"
    ]
    has_model = any(os.path.exists(f) for f in all_model_files)

    if has_model:
        print("✅ 检测到已有模型文件，开始加载...")
        predictor.load_models()
    else:
        print("⚠️ 未检测到模型，开始完整训练（基模型 + Stacking堆叠模型）")
        CSV_FILE = "电影数据.csv"
        if os.path.exists(CSV_FILE):
            X, y, le, mlb = load_and_preprocess(CSV_FILE)
            predictor.le = le
            predictor.mlb = mlb
            predictor.train_with_kfold(X, y, n_splits=5)
            predictor.train_stacking(X, y)
            predictor.save_models()
            print("✅ 全部模型训练+保存完成！")
        else:
            print("❌ 未找到 电影数据.csv，进入模拟预测模式")

    print("🚀 启动网页应用...")
    run_streamlit_app(predictor)