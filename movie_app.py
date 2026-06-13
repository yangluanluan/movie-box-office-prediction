# 依赖安装（首次运行执行）
# pip install streamlit pandas numpy scikit-learn joblib matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import joblib
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ===================== 强制加载中文字体（Streamlit 云环境专用）=====================
# 不再依赖 rcParams，直接加载字体文件，绘图时强制指定
def get_chinese_font():
    # Streamlit 云环境自带的文泉驿字体路径（固定不变）
    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    # 本地环境 fallback
    if sys.platform.startswith("win"):
        return fm.FontProperties(fname="C:/Windows/Fonts/simhei.ttf")
    elif sys.platform == "darwin":
        return fm.FontProperties(fname="/Library/Fonts/PingFang.ttc")
    # 兜底方案
    return None

# 全局字体对象，后续所有绘图都用它
chinese_font = get_chinese_font()
plt.rcParams['axes.unicode_minus'] = False

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
        self.stacking_test_rmse = stacking_rmse
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
        joblib.dump(self.stacking_test_rmse, "stacking_rmse.pkl")
        joblib.dump(self.model_scores, "model_scores.pkl")
        print("\n✅ 所有模型、性能指标和特征配置已保存")

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
        scores_path = "model_scores.pkl"
        if os.path.exists(scores_path):
            self.model_scores = joblib.load(scores_path)
            print("✅ 模型交叉验证性能指标加载成功")
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
    le = LabelEncoder()
    X['certificate'] = le.fit_transform(X['certificate'].astype(str))
    X['DIRECTOR'] = le.fit_transform(X['DIRECTOR'].astype(str))
    X['ACTOR 1'] = le.fit_transform(X['ACTOR 1'].astype(str))
    X['ACTOR 2'] = le.fit_transform(X['ACTOR 2'].astype(str))

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

    # 票房预测页
    elif choice == "票房预测":
        st.title("电影票房预测模型")
        st.subheader("各机器学习模型预测性能对比（RMSE）")
        if predictor.model_scores and predictor.stacking_test_rmse is not None:
            model_names_cn = [
                "线性回归",
                "梯度提升回归",
                "Ridge 回归",
                "Lasso回归",
                "随机森林回归",
                "模型融合"
            ]
            try:
                lr_rmse = predictor.model_scores["linear_regression"]["mean_cv_rmse"]
                gbr_rmse = predictor.model_scores["gradient_boosting"]["mean_cv_rmse"]
                ridge_rmse = predictor.model_scores["ridge_regression"]["mean_cv_rmse"]
                lasso_rmse = predictor.model_scores["lasso_regression"]["mean_cv_rmse"]
                rf_rmse = predictor.model_scores["random_forest"]["mean_cv_rmse"]
                stack_rmse = predictor.stacking_test_rmse

                rmse_data = [lr_rmse, gbr_rmse, ridge_rmse, lasso_rmse, rf_rmse, stack_rmse]

                # ========== 关键修改：绘图时强制指定中文字体 ==========
                fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
                bars = ax.bar(model_names_cn, rmse_data, color='#642EFE')
                
                # 所有中文元素都加上 fontproperties=chinese_font
                ax.set_title("机器学习电影票房预测性能对比", fontsize=14, fontproperties=chinese_font)
                ax.set_ylabel("rmse", fontproperties=chinese_font)
                ax.set_xlabel("model", fontproperties=chinese_font)
                plt.xticks(fontproperties=chinese_font)  # X轴标签强制指定字体
                plt.yticks(fontproperties=chinese_font)  # Y轴标签也加上

                offset = max(rmse_data) * 0.01
                for bar, val in zip(bars, rmse_data):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + offset,
                        f"{val:.4f}",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontproperties=chinese_font  # 柱子上的数字也加上
                    )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.info("提示：RMSE数值越低，模型预测误差越小，预测效果越好。模型融合（Stacking）通常具备最优性能。")
            except KeyError:
                st.warning("模型性能指标不完整，请删除所有.pkl文件后重新完整训练模型！")
        else:
            st.warning("暂无训练完成的模型性能数据，请先运行模型训练流程生成模型文件！")
        st.divider()

        # 原有预测表单代码不变
        df = pd.read_csv("电影数据.csv")
        df.columns = [c.strip() for c in df.columns]

        directors = sorted(df['DIRECTOR'].dropna().unique().tolist())
        actors1 = sorted(df['ACTOR 1'].dropna().unique().tolist())
        actors2 = sorted(df['ACTOR 2'].dropna().unique().tolist())

        model_map = {
            "多元线性回归模型": "linear_regression",
            "Ridge回归模型": "ridge_regression",
            "Lasso回归模型": "lasso_regression",
            "随机森林回归模型": "random_forest",
            "决策树回归模型": "gradient_boosting",
            "Stacking融合模型": "stacking_model"
        }
        select_cn = st.selectbox("请选择预测模型", list(model_map.keys()))
        select_model = model_map[select_cn]

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
            submit = st.form_submit_button("开始预测")

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
                st.success(f"当前使用模型：{select_cn}")
                st.markdown(f"### 预测票房：:red[{pred_real:.0f}] 元")
            except Exception as e:
                st.error(f"预测异常：{str(e)}，启用模拟结果")
                pred = 371300633
                st.markdown(f"### 模拟预测票房：:red[{pred}] 万元")

    # 票房分析页
    elif choice == "票房分析":
        st.title("📊 电影票房数据分析")
        st.write("以下是电影数据的可视化分析图表，帮助你理解票房的影响因素")

        df = pd.read_csv("电影数据.csv")
        df.columns = [c.strip() for c in df.columns]
        df['GROSS COLLECTION'] = df['GROSS COLLECTION'].apply(parse_gross)
        df = df.dropna(subset=['GROSS COLLECTION']).reset_index(drop=True)
        df['Year'] = df['Year'].apply(extract_year)
        df['runtime'] = df.astype(str)['runtime'].str.replace(' min', '').astype(float)
        df['votes'] = df.astype(str)['votes'].str.replace(',', '').astype(float)

        # 1. 年份票房趋势
        st.subheader("1. 电影票房随年份变化趋势：呈现出长期增长、阶段性波动、头部效应加剧的趋势")
        year_gross = df.groupby('Year')['GROSS COLLECTION'].mean().dropna()
        st.line_chart(year_gross)

        # 2. 评分与票房
        st.subheader("2. 电影评分与票房关系：评分与票房之间存在弱正相关关系，票房较高的电影普遍集中在 7.5-8.5 分的中等偏上区间，而评分超过 9.0 分的电影票房上限反而有所下降")
        scatter_df = df[['RATING', 'GROSS COLLECTION']].dropna()
        st.scatter_chart(scatter_df, x='RATING', y='GROSS COLLECTION')

        # 3. 电影类型平均票房
        st.subheader("3. 不同类型电影的平均票房：冒险类影片平均票房最高")
        def split_genre(x):
            return str(x).split(',')
        df['genre_list'] = df['genre'].apply(split_genre)
        genre_gross = {}
        for idx, row in df.iterrows():
            gross = row['GROSS COLLECTION']
            for g in row['genre_list']:
                if g not in genre_gross:
                    genre_gross[g] = []
                genre_gross[g].append(gross)
        genre_avg = {k: np.mean(v) for k, v in genre_gross.items() if k != "nan"}
        genre_df = pd.DataFrame(list(genre_avg.items()), columns=['类型', '平均票房'])
        st.bar_chart(genre_df.set_index('类型'))

        # 4. 导演Top10
        st.subheader("4. 导演Top10平均票房：Jon Watts导演作品平均票房最高")
        director_gross = df.groupby('DIRECTOR')['GROSS COLLECTION'].mean().sort_values(ascending=False).head(10)
        st.bar_chart(director_gross)

        # 5. 电影时长 & 票房关系
        st.subheader("5. 电影时长 与 票房收入关系：100–180 分钟为票房黄金区间")
        runtime_gross_df = df[['runtime', 'GROSS COLLECTION']].dropna()
        st.scatter_chart(runtime_gross_df, x='runtime', y='GROSS COLLECTION')

        # 6. MPAA分级整体数量分布
        st.subheader("6. MPAA电影分级整体数量分布")
        mpaa_count = df['certificate'].value_counts().dropna()
        st.bar_chart(mpaa_count)

        # 7. MPAA分级 与 电影时长分布（强制指定字体）
        st.subheader("7. MPAA分级 与 电影时长分布：大部分影片时长集中在110–135分钟")
        mpaa_runtime_df = df[['certificate', 'runtime']].dropna()
        fig, ax = plt.subplots(figsize=(10, 6))
        mpaa_runtime_df.boxplot(by='certificate', column='runtime', ax=ax)
        ax.set_title('MPAA Rating vs Movie Runtime Distribution', fontproperties=chinese_font)
        ax.set_xlabel('MPAA Rating', fontproperties=chinese_font)
        ax.set_ylabel('Movie Runtime (Minutes)', fontproperties=chinese_font)
        plt.xticks(fontproperties=chinese_font)
        plt.yticks(fontproperties=chinese_font)
        plt.suptitle('')
        st.pyplot(fig)
        plt.close(fig)

        # 8. MPAA分级 与 电影票房收入分布（强制指定字体）
        st.subheader("8. MPAA分级 与 电影票房收入分布：12A分级影片整体票房表现最优")
        mpaa_gross_df = df[['certificate', 'GROSS COLLECTION']].dropna()
        fig, ax = plt.subplots(figsize=(10, 6))
        mpaa_gross_df.boxplot(by='certificate', column='GROSS COLLECTION', ax=ax)
        ax.set_title('MPAA Rating vs Movie Box Office Distribution', fontproperties=chinese_font)
        ax.set_xlabel('MPAA Rating', fontproperties=chinese_font)
        ax.set_ylabel('Box Office Revenue (USD)', fontproperties=chinese_font)
        plt.xticks(fontproperties=chinese_font)
        plt.yticks(fontproperties=chinese_font)
        plt.suptitle('')
        st.pyplot(fig)
        plt.close(fig)

        st.info("💡 提示：所有图表数据均来自电影数据集，可根据需要拓展分析维度")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    predictor = MovieBoxOfficePredictor()

    all_model_files = [
        "linear_regression.pkl",
        "ridge_regression.pkl",
        "lasso_regression.pkl",
        "random_forest.pkl",
        "gradient_boosting.pkl",
        "stacking_model.pkl",
        "model_scores.pkl"
    ]
    has_model = any(os.path.exists(f) for f in all_model_files)

    CSV_FILE = "电影数据.csv"
    if has_model:
        print("✅ 检测到已有模型，直接加载...")
        predictor.load_models()
    else:
        if os.path.exists(CSV_FILE):
            print("⚠️ 无预训练模型，开始训练模型，请稍等...")
            X, y, le, mlb = load_and_preprocess(CSV_FILE)
            predictor.le = le
            predictor.mlb = mlb
            predictor.train_with_kfold(X, y)
            predictor.train_stacking(X, y)
            predictor.save_models()
            print("✅ 模型训练并保存完成！")
        else:
            print("❌ 未找到 电影数据.csv，进入模拟预测模式")

    print("🚀 启动电影票房系统...")
    run_streamlit_app(predictor)
