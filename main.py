import logging, time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_process.error_inject import inject_errors
from data_process.id_align import align_id
from fix.oriclean import oriclean
from fix.fix_ori import RepairPredictor, unbiased_generate_repairs
from select.master_select import master_select
from SV_estimate.SHAP_cal import estimate_shapley_with_shap
# ← 新 estimator（A 或 B 方案都在同一路径）
from SV_estimate.SV_est_mont import estimate_cluster_shapley_mc
import warnings
from sklearn.exceptions import ConvergenceWarning

# 禁用 LogisticRegression 的收敛警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# -------------------- 日志配置 --------------------
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/run.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# -------------------- 参数 --------------------
master_size = 0.8
data_input = "data/iris.data"
error_num = 0.9
n_mc = 30              # Monte‑Carlo 轮数（30 足够稳定且很快）


# -------------------- 进度包装 --------------------
def shapley_with_progress(X, y, repair_dict, num_samples):
    """在 tqdm 里分块调用 estimator，以显示采样进度"""
    chunk = max(1, num_samples // 10)
    shapley = {i: 0.0 for i in range(len(X))}
    done = 0
    with tqdm(total=num_samples, desc="MC 采样", unit="sample") as pbar:
        while done < num_samples:
            step = min(chunk, num_samples - done)
            tmp = estimate_cluster_shapley_mc(
                X, y, repair_dict, num_samples=step, rng_seed=int(time.time()*1e6)%2**32
            )
            for k, v in tmp.items():
                shapley[k] += v * (step / num_samples)
            done += step
            pbar.update(step)
    return shapley


# -------------------- 主流程 --------------------
def main():
    t0 = time.time()

    # 1. 读取干净数据
    data = pd.read_csv(data_input, header=None)
    if data.shape[1] == 5:
        data["id"] = range(len(data))
    clean_data = data.copy()
    n_cols = clean_data.shape[1]
    feat_cols = list(range(n_cols - 2)); label_col = n_cols - 2
    log.info(f"Loaded data: {data.shape}")

    # 2. id 对齐
    id_data = align_id(data_input)

    # 3. 注错
    error_data, error_info = inject_errors(
        id_data, error_rate=error_num,
        id_column=id_data.columns[-1], label_column=id_data.columns[-2]
    )
    log.info(f"Injected {len(error_info)} errors.")

    # 4. 粗糙修复
    oriclean_data = oriclean(error_data, error_info)

    # 5. φ_rough via SHAP
    base_model = RandomForestClassifier(random_state=42)
    base_model.fit(clean_data.iloc[:, feat_cols], clean_data.iloc[:, label_col])
    raw_shapley = {i: v for i, v in enumerate(
        estimate_shapley_with_shap(base_model, clean_data.iloc[:, feat_cols].values)
    )}
    log.info("Computed φ_rough.")

    # 6. 粗修准确率日志
    X_train, X_test, y_train, y_test = train_test_split(
        oriclean_data.iloc[:, feat_cols], clean_data.iloc[:, label_col],
        test_size=0.2, random_state=42)
    tmp_acc = accuracy_score(y_test,
        RandomForestClassifier(random_state=42).fit(X_train, y_train).predict(X_test))
    log.info(f"Rough‑repair accuracy: {tmp_acc:.4f}")

    # 7. 生成修复候选
    predictor = RepairPredictor()
    predictor.register_method("unbiased", unbiased_generate_repairs)
    possible_repairs = predictor.generate_repairs(
        data=error_data,
        error_info=tqdm(error_info, desc="Repair candidates"),
        method_name="unbiased",
        clean_data=clean_data,
        numeric_indices=feat_cols
    )
    log.info("Generated repair candidates.")

    # 8. φ_refined via cluster‑MC (新接口)
    log.info("Estimating φ_refined (cluster‑MC)...")
    true_shap = shapley_with_progress(
        error_data.iloc[:, feat_cols].values,
        clean_data.iloc[:, label_col].values,
        possible_repairs,
        num_samples=n_mc
    )
    log.info("φ_refined estimation done.")

    # 9. 主数据一次性挑选
    combined = {i: true_shap[i] + (true_shap[i] - raw_shapley.get(i, 0.0))
                for i in true_shap}
    master_data, selected = master_select(
        sv=combined, clean_data=clean_data,
        dirty_data=error_data, select_num=master_size
    )
    log.info(f"Selected {len(selected)} rows for perfect repair.")

    # 10. 最终评估
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        master_data.iloc[:, feat_cols], clean_data.iloc[:, label_col],
        test_size=0.2, random_state=42)
    final_acc = accuracy_score(y_test_m,
        RandomForestClassifier(random_state=42).fit(X_train_m, y_train_m).predict(X_test_m))
    log.info(f"Final accuracy after perfect repair: {final_acc:.4f}")
    log.info(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
