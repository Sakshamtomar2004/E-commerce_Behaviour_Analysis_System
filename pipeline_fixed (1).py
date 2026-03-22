# =============================================================================
# E-Commerce Purchase Prediction — GRU + Hybrid Pipeline
# =============================================================================
# WHAT THIS FILE DOES:
#   1. Loads ml_dataframe from output_Xgboost/final_ml_dataframe.csv (already made)
#   2. Loads the best XGBoost model from output_Xgboost/ (already trained)
#   3. Builds GRU sequence data from raw CSVs
#   4. Trains GRU with strong anti-overfitting regularization
#   5. Extracts GRU embeddings → trains Hybrid XGBoost
#   6. Final 3-model comparison (XGBoost vs GRU vs Hybrid)

import matplotlib
matplotlib.use('Agg')  # no popup windows — saves all plots to disk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc, time, json, os, pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve, auc,
    f1_score, recall_score, roc_auc_score,
    precision_recall_curve
)
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, Masking,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================================================================
# PATHS — edit these if your folder structure is different
# =============================================================================
XGB_OUTPUT_DIR  = r'C:\Users\UIET\Desktop\Sahil ML\output_Xgboost'              # where XGBoost outputs live
GRU_OUTPUT_DIR  = 'output_GRU_Hybrid'            # where this script saves outputs
ML_DATAFRAME_CSV = os.path.join(XGB_OUTPUT_DIR, 'final_ml_dataframe.csv')

# Raw dataset CSVs (needed for GRU sequence building)
PATH_OCT = r'ecommerce-behavior-data-from-multi-category-store\2019-Oct.csv'
PATH_NOV = r'ecommerce-behavior-data-from-multi-category-store\2019-Nov.csv'

os.makedirs(GRU_OUTPUT_DIR, exist_ok=True)
sns.set_theme(style='whitegrid')

# =============================================================================
# GPU CONFIGURATION
# =============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'GPU(s) configured: {[g.name for g in gpus]}')
    except RuntimeError as e:
        print(f'GPU config error: {e}')
else:
    print('No GPU found — running on CPU')

print(f'TensorFlow : {tf.__version__}')
print(f'NumPy      : {np.__version__}')
print(f'Pandas     : {pd.__version__}')


# =============================================================================
# STEP 1 — Load ml_dataframe (from XGBoost preprocessing output)
# =============================================================================
print('\n' + '='*65)
print('STEP 1: LOADING ml_dataframe FROM CACHE')
print('='*65)

if not os.path.exists(ML_DATAFRAME_CSV):
    raise FileNotFoundError(
        f'Cannot find {ML_DATAFRAME_CSV}\n'
        'Please run Preprocessing_XGboost.py first to generate this file.'
    )

ml_dataframe = pd.read_csv(ML_DATAFRAME_CSV, index_col='user_session')

# Drop any stray Unnamed columns from CSV round-trip
ml_dataframe = ml_dataframe.loc[:, ~ml_dataframe.columns.str.contains('^Unnamed')]

print(f'Loaded ml_dataframe shape : {ml_dataframe.shape}')
print(f'Columns : {ml_dataframe.columns.tolist()}')

vc = ml_dataframe['target_purchase'].value_counts()
print(f'\nTarget distribution:')
print(f'  Non-buyers (0): {vc[0]:,}  ({vc[0]/len(ml_dataframe)*100:.2f}%)')
print(f'  Buyers     (1): {vc[1]:,}  ({vc[1]/len(ml_dataframe)*100:.2f}%)')


# =============================================================================
# STEP 2 — Load best XGBoost model (compare set_1 vs set_2, pick winner)
# =============================================================================
print('\n' + '='*65)
print('STEP 2: LOADING BEST XGBOOST MODEL')
print('='*65)

def load_xgb_model_and_metrics(set_num):
    model_path   = os.path.join(XGB_OUTPUT_DIR, f'champion_xgb_model_set_{set_num}.pkl')
    metrics_path = os.path.join(XGB_OUTPUT_DIR, f'evaluation_metrics_set_{set_num}.json')
    if not os.path.exists(model_path) or not os.path.exists(metrics_path):
        return None, None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return model, metrics

model_1, metrics_1 = load_xgb_model_and_metrics(1)
model_2, metrics_2 = load_xgb_model_and_metrics(2)

FEATURE_SET_1 = ['total_carts', 'unique_products', 'session_duration_seconds', 'focus_ratio']
FEATURE_SET_2 = ['total_carts', 'views_per_minute', 'focus_ratio', 'price_exploration_capped']

if metrics_1 is None and metrics_2 is None:
    raise FileNotFoundError(
        f'No XGBoost models found in {XGB_OUTPUT_DIR}/\n'
        'Please run XG_Boost.py first.'
    )

# Pick the set with higher ROC-AUC
roc_1 = metrics_1['roc_auc'] if metrics_1 else -1
roc_2 = metrics_2['roc_auc'] if metrics_2 else -1

if roc_1 >= roc_2:
    xgb_champion   = model_1
    xgb_features   = FEATURE_SET_1
    xgb_set_winner = 1
    xgb_metrics    = metrics_1
else:
    xgb_champion   = model_2
    xgb_features   = FEATURE_SET_2
    xgb_set_winner = 2
    xgb_metrics    = metrics_2

print(f'Set 1 ROC-AUC : {roc_1:.4f}  Features: {FEATURE_SET_1}')
print(f'Set 2 ROC-AUC : {roc_2:.4f}  Features: {FEATURE_SET_2}')
print(f'\nWinner → Set {xgb_set_winner}  (ROC-AUC: {max(roc_1, roc_2):.4f})')
print(f'XGBoost champion loaded. Features used: {xgb_features}')

# Reconstruct XGBoost test predictions for final comparison
# (using the saved y_test and y_pred arrays from XG_Boost.py)
y_test_xgb_path  = os.path.join(XGB_OUTPUT_DIR, f'y_test_set_{xgb_set_winner}.npy')
y_pred_xgb_path  = os.path.join(XGB_OUTPUT_DIR, f'y_pred_set_{xgb_set_winner}.npy')
y_probs_xgb_path = os.path.join(XGB_OUTPUT_DIR, f'y_probs_set_{xgb_set_winner}.npy')

if os.path.exists(y_test_xgb_path):
    xgb_y_test  = np.load(y_test_xgb_path)
    xgb_y_pred  = np.load(y_pred_xgb_path)
    xgb_y_probs = np.load(y_probs_xgb_path)
    xgb_auc     = roc_auc_score(xgb_y_test, xgb_y_probs)
    fpr_xgb, tpr_xgb, _ = roc_curve(xgb_y_test, xgb_y_probs)
    print(f'XGBoost saved predictions loaded. AUC = {xgb_auc:.4f}')
else:
    # Fallback: recompute on ml_dataframe if saved arrays don't exist
    print('Saved prediction arrays not found — recomputing on ml_dataframe...')
    X_full = ml_dataframe.drop(columns=['target_purchase'])
    y_full = ml_dataframe['target_purchase']
    _, xgb_X_test, _, xgb_y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    xgb_X_test_f = xgb_X_test[xgb_features].astype('float32')
    xgb_y_pred   = xgb_champion.predict(xgb_X_test_f)
    xgb_y_probs  = xgb_champion.predict_proba(xgb_X_test_f)[:, 1]
    xgb_auc      = roc_auc_score(xgb_y_test, xgb_y_probs)
    fpr_xgb, tpr_xgb, _ = roc_curve(xgb_y_test, xgb_y_probs)
    print(f'XGBoost recomputed. AUC = {xgb_auc:.4f}')


# =============================================================================
# STEP 3 — Load raw CSVs for GRU sequence building
# =============================================================================
print('\n' + '='*65)
print('STEP 3: LOADING RAW EVENT DATA FOR GRU SEQUENCES')
print('='*65)

columns_to_keep = [
    'event_time', 'event_type', 'product_id',
    'category_code', 'brand', 'price',
    'user_id', 'user_session'
]

print('Reading October...')
df_oct = pd.read_csv(PATH_OCT, usecols=columns_to_keep)
print(f'  Oct shape : {df_oct.shape}')

print('Reading November...')
df_nov = pd.read_csv(PATH_NOV, usecols=columns_to_keep)
print(f'  Nov shape : {df_nov.shape}')

final_df = pd.concat([df_oct, df_nov], ignore_index=True)
del df_oct, df_nov
gc.collect()

final_df = final_df.drop_duplicates()
final_df = final_df[final_df['price'] > 0].copy()
final_df['event_time'] = pd.to_datetime(final_df['event_time'])
print(f'Raw event data ready: {final_df.shape}')


# =============================================================================
# STEP 4 — Build GRU Sequence Data
# =============================================================================
print('\n' + '='*65)
print('STEP 4: BUILDING GRU SEQUENCE DATA')
print('='*65)

seq_df = final_df.sort_values(by=['user_session', 'event_time']).copy()

session_targets = (
    seq_df.groupby('user_session')['event_type']
    .apply(lambda x: 1 if 'purchase' in x.values else 0)
    .to_dict()
)
print(f'Sessions with purchase  : {sum(session_targets.values()):,}')
print(f'Sessions without purchase: {sum(1 for v in session_targets.values() if v == 0):,}')

# CRITICAL: remove purchase events from input — prevents label leakage
seq_df = seq_df[seq_df['event_type'] != 'purchase'].copy()

# Price: log1p then MinMaxScale
seq_df['log_price']    = np.log1p(seq_df['price'])
price_scaler           = MinMaxScaler()
seq_df['scaled_price'] = price_scaler.fit_transform(seq_df[['log_price']])

import pickle as pkl
with open(os.path.join(GRU_OUTPUT_DIR, 'price_scaler.pkl'), 'wb') as f:
    pkl.dump(price_scaler, f)
print('Saved: price_scaler.pkl')

seq_df['is_view'] = (seq_df['event_type'] == 'view').astype(int)
seq_df['is_cart'] = (seq_df['event_type'] == 'cart').astype(int)

seq_df['time_gap_seconds'] = (
    seq_df.groupby('user_session')['event_time']
    .diff().dt.total_seconds().fillna(0)
)
seq_df['log_time_gap'] = np.log1p(seq_df['time_gap_seconds'])

sequence_features = ['is_view', 'is_cart', 'scaled_price', 'log_time_gap']
session_groups = (
    seq_df.groupby('user_session')[sequence_features]
    .apply(lambda x: x.values.tolist())
)

gru_y           = np.array([session_targets[s] for s in session_groups.index])
gru_session_ids = np.array(session_groups.index)

MAX_STEPS = 15
gru_X = pad_sequences(
    session_groups.tolist(),
    maxlen=MAX_STEPS,
    padding='post',
    truncating='pre',
    dtype='float32'
)
gru_X = np.nan_to_num(gru_X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
gru_y = gru_y.astype(np.float32)

print(f'\nGRU input shape : {gru_X.shape}  →  (sessions, time_steps, features)')
print(f'GRU target shape: {gru_y.shape}')
print(f'Purchase rate   : {gru_y.mean()*100:.2f}%')

lengths = seq_df.groupby('user_session').size()
print(f'\nSession length percentiles:')
for p in [50, 75, 90, 95, 99]:
    print(f'  p{p}: {lengths.quantile(p/100):.0f} events')

# Save arrays so you never have to rebuild them
np.save(os.path.join(GRU_OUTPUT_DIR, 'gru_X.npy'), gru_X)
np.save(os.path.join(GRU_OUTPUT_DIR, 'gru_y.npy'), gru_y)
np.save(os.path.join(GRU_OUTPUT_DIR, 'gru_session_ids.npy'), gru_session_ids)
print('Saved: gru_X.npy, gru_y.npy, gru_session_ids.npy')


# =============================================================================
# STEP 5 — Train/Test Split for GRU
# =============================================================================
gru_X_train, gru_X_test, gru_y_train, gru_y_test, gru_sess_train, gru_sess_test = \
    train_test_split(
        gru_X, gru_y, gru_session_ids,
        test_size=0.2, random_state=42, stratify=gru_y
    )

neg_count = (gru_y_train == 0).sum()
pos_count = (gru_y_train == 1).sum()
class_weight_dict = {0: 1.0, 1: float(neg_count / pos_count)}

print(f'\nGRU train : {len(gru_X_train):,} sessions')
print(f'GRU test  : {len(gru_X_test):,} sessions')
print(f'Class weight for buyers: {class_weight_dict[1]:.2f}')


# =============================================================================
# STEP 6 — Build & Train GRU (Anti-Overfitting Architecture)
# =============================================================================
# HOW OVERFITTING IS PREVENTED:
#   1. recurrent_dropout=0.2  — drops 20% of recurrent connections each step
#   2. Dropout(0.5)           — heavy dropout before Dense layers
#   3. BatchNormalization      — stabilizes activations, reduces internal covariate shift
#   4. L2 regularization      — penalizes large weights in Dense layers
#   5. gradient_clipnorm=1.0  — prevents exploding gradients
#   6. ReduceLROnPlateau       — halves LR if val_auc stalls for 2 epochs
#   7. EarlyStopping(patience=6) — stops if val_auc doesn't improve for 6 epochs
#   8. ModelCheckpoint         — only saves the best epoch, never a worse one
# =============================================================================
print('\n' + '='*65)
print('STEP 6: BUILDING & TRAINING ANTI-OVERFIT GRU MODEL')
print('='*65)

inputs      = Input(shape=(MAX_STEPS, 4), name='sequence_input')
x           = Masking(mask_value=0.0)(inputs)

# GRU Layer 1 — detects short-term patterns
x = GRU(
    64,
    return_sequences=True,
    activation='tanh',
    recurrent_dropout=0.2,      
    kernel_regularizer=l2(1e-4),
    name='gru_layer_1'
)(x)
x = BatchNormalization()(x)     # stabilize layer 1 output

# GRU Layer 2 — 32-dim session fingerprint (embedding for hybrid)
gru_emb_out = GRU(
    32,
    return_sequences=False,
    activation='tanh',
    recurrent_dropout=0.2,
    kernel_regularizer=l2(1e-4),
    name='gru_embedding_layer'
)(x)
x = BatchNormalization()(gru_emb_out)

x      = Dropout(0.5)(x)                                   # heavy dropout
x      = Dense(32, activation='relu',
               kernel_regularizer=l2(1e-4))(x)             # L2 on dense
x      = Dropout(0.3)(x)                                   # second dropout
output = Dense(1, activation='sigmoid', name='purchase_output')(x)

gru_model = Model(inputs=inputs, outputs=output, name='GRU_Classifier')
gru_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0           # gradient clipping — prevents exploding gradients
    ),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
gru_model.summary()

callbacks = [
    # Save best model by val_auc
    ModelCheckpoint(
        filepath=os.path.join(GRU_OUTPUT_DIR, 'gru_best_model.keras'),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    # Stop if val_auc doesn't improve for 6 epochs
    EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    # Halve LR if val_auc stalls for 2 epochs
    ReduceLROnPlateau(
        monitor='val_auc',
        mode='max',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

print('\nTraining GRU (anti-overfit mode)...')
t0 = time.time()
gru_history = gru_model.fit(
    gru_X_train, gru_y_train,
    epochs=30,                   # more epochs — early stopping handles the rest
    batch_size=512,
    validation_split=0.2,        # 20% of train used for val monitoring
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)
print(f'GRU training completed in {(time.time()-t0)/60:.1f} min')

gru_model.save_weights(os.path.join(GRU_OUTPUT_DIR, 'gru_final_weights.weights.h5'))
print('Saved: gru_best_model.keras')
print('Saved: gru_final_weights.weights.h5')


# =============================================================================
# STEP 7 — GRU Evaluation & Training Curves
# =============================================================================
print('\n' + '='*65)
print('STEP 7: GRU EVALUATION')
print('='*65)

gru_y_probs = gru_model.predict(gru_X_test, batch_size=1024).flatten()

# F2 threshold: recall weighted 2x over precision
# Missing a buyer (FN) costs more than false alarm (FP)
precisions_g, recalls_g, thresholds_g = precision_recall_curve(gru_y_test, gru_y_probs)
beta = 2
f2_g = np.where(
    (beta**2 * precisions_g + recalls_g) > 0,
    (1 + beta**2) * precisions_g * recalls_g / (beta**2 * precisions_g + recalls_g),
    0
)
best_thresh_g = thresholds_g[np.argmax(f2_g[:-1])]
gru_y_pred    = (gru_y_probs >= best_thresh_g).astype(int)
gru_auc       = roc_auc_score(gru_y_test, gru_y_probs)

print(f'Optimal F2 threshold : {best_thresh_g:.4f}')
print(f'Accuracy  : {accuracy_score(gru_y_test, gru_y_pred)*100:.2f}%')
print(f'Recall    : {recall_score(gru_y_test, gru_y_pred)*100:.2f}%')
print(f'F1 Score  : {f1_score(gru_y_test, gru_y_pred)*100:.2f}%')
print(f'ROC-AUC   : {gru_auc:.4f}')
print()
print(classification_report(gru_y_test, gru_y_pred, target_names=['No Purchase', 'Purchase']))

# ── TRAINING CURVES ──────────────────────────────────────────────────────────
auc_key     = [k for k in gru_history.history if 'auc' in k and 'val' not in k][0]
val_auc_key = 'val_' + auc_key

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

axes[0].plot(gru_history.history['loss'],        label='Train Loss', color='#4C72B0', lw=2)
axes[0].plot(gru_history.history['val_loss'],     label='Val Loss',   color='#DD8452', lw=2, ls='--')
axes[0].set_title('GRU — Loss Curve', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Binary Crossentropy')
axes[0].legend()

axes[1].plot(gru_history.history['accuracy'],     label='Train Acc', color='#4C72B0', lw=2)
axes[1].plot(gru_history.history['val_accuracy'], label='Val Acc',   color='#DD8452', lw=2, ls='--')
axes[1].set_title('GRU — Accuracy Curve', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend()

axes[2].plot(gru_history.history[auc_key],        label='Train AUC', color='#4C72B0', lw=2)
axes[2].plot(gru_history.history[val_auc_key],    label='Val AUC',   color='#DD8452', lw=2, ls='--')
axes[2].set_title('GRU — AUC Curve', fontweight='bold', fontsize=13)
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('AUC')
axes[2].legend()

plt.suptitle('GRU Training Curves (Anti-Overfit)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRU_OUTPUT_DIR, 'gru_training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: gru_training_curves.png')

# ── GRU CONFUSION MATRIX ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_g = confusion_matrix(gru_y_test, gru_y_pred)
sns.heatmap(cm_g, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Purchase', 'Purchase'],
            yticklabels=['No Purchase', 'Purchase'], ax=axes[0])
axes[0].set_title('GRU — Confusion Matrix', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

fpr_g, tpr_g, _ = roc_curve(gru_y_test, gru_y_probs)
axes[1].plot(fpr_g, tpr_g, color='#DD8452', lw=2, label=f'GRU AUC = {gru_auc:.3f}')
axes[1].plot([0,1],[0,1],'k--',lw=1,label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('GRU — ROC-AUC', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(GRU_OUTPUT_DIR, 'gru_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: gru_results.png')


# =============================================================================
# STEP 8 — Extract 32-dim GRU Embeddings for Hybrid Model
# =============================================================================
print('\n' + '='*65)
print('STEP 8: EXTRACTING GRU EMBEDDINGS (32-dim per session)')
print('='*65)

# Freeze GRU — cut model right after gru_embedding_layer
embedding_extractor = Model(
    inputs=gru_model.input,
    outputs=gru_model.get_layer('gru_embedding_layer').output,
    name='GRU_Embedding_Extractor'
)
embedding_extractor.trainable = False

print('Extracting train embeddings...')
train_embeddings = embedding_extractor.predict(gru_X_train, batch_size=1024, verbose=1)
print('Extracting test embeddings...')
test_embeddings  = embedding_extractor.predict(gru_X_test,  batch_size=1024, verbose=1)

print(f'\nTrain embedding shape : {train_embeddings.shape}')
print(f'Test  embedding shape : {test_embeddings.shape}')

emb_cols     = [f'gru_emb_{i}' for i in range(32)]
train_emb_df = pd.DataFrame(train_embeddings, columns=emb_cols, index=gru_sess_train)
test_emb_df  = pd.DataFrame(test_embeddings,  columns=emb_cols, index=gru_sess_test)

print('\nSample embedding (3 sessions, first 6 dims):')
print(train_emb_df.iloc[:3, :6].round(4))


# =============================================================================
# STEP 9 — Align + Concatenate GRU embeddings with tabular features
# =============================================================================
print('\n' + '='*65)
print('STEP 9: BUILDING HYBRID FEATURE MATRIX')
print('='*65)

tab_features = ml_dataframe.drop(columns=['target_purchase'])

# inner join — only sessions present in BOTH embedding df and ml_dataframe
hybrid_train = train_emb_df.join(tab_features, how='inner')
hybrid_test  = test_emb_df.join(tab_features,  how='inner')

hybrid_y_train = ml_dataframe.loc[hybrid_train.index, 'target_purchase']
hybrid_y_test  = ml_dataframe.loc[hybrid_test.index,  'target_purchase']

n_gru_dims = len(emb_cols)
n_tab_dims = tab_features.shape[1]

print(f'Hybrid train shape : {hybrid_train.shape}')
print(f'Hybrid test  shape : {hybrid_test.shape}')
print(f'Feature breakdown  : {n_gru_dims} GRU dims + {n_tab_dims} tabular = {n_gru_dims + n_tab_dims} total')
print(f'Train target dist  : {hybrid_y_train.value_counts().to_dict()}')


# =============================================================================
# STEP 10 — Train Hybrid XGBoost
# Early stopping via eval_set + early_stopping_rounds
# =============================================================================
print('\n' + '='*65)
print('STEP 10: TRAINING HYBRID XGBOOST (GRU Embeddings + Tabular)')
print('='*65)

hybrid_imbalance = (hybrid_y_train == 0).sum() / (hybrid_y_train == 1).sum()
print(f'Hybrid scale_pos_weight : {hybrid_imbalance:.2f}')

# Convert to float32 for GPU
hybrid_train_f = hybrid_train.astype('float32')
hybrid_test_f  = hybrid_test.astype('float32')

xgb_hybrid = XGBClassifier(
    scale_pos_weight  = hybrid_imbalance,
    n_estimators      = 500,        # high cap — early stopping handles it
    learning_rate     = 0.05,
    max_depth         = 6,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    gamma             = 0.5,
    reg_alpha         = 0.1,        # L1 regularization
    reg_lambda        = 1.0,        # L2 regularization
    random_state      = 42,
    n_jobs            = 1,
    tree_method       = 'hist',
    device            = 'cuda',     # uses your GPU
    eval_metric       = 'auc',
    early_stopping_rounds = 20,     # stop if AUC doesn't improve for 20 rounds
    verbosity         = 0
)

t0 = time.time()
xgb_hybrid.fit(
    hybrid_train_f, hybrid_y_train,
    eval_set=[(hybrid_test_f, hybrid_y_test)],
    verbose=50
)
best_iter = xgb_hybrid.best_iteration
print(f'\nHybrid XGBoost trained in {(time.time()-t0)/60:.1f} min')
print(f'Best iteration (early stopping) : {best_iter}')

model_path = os.path.join(GRU_OUTPUT_DIR, 'xgboost_hybrid.pkl')
with open(model_path, 'wb') as f:
    pkl.dump(xgb_hybrid, f)
print(f'Saved: {model_path}')


# =============================================================================
# STEP 11 — Hybrid Model Evaluation & Plots
# =============================================================================
print('\n' + '='*65)
print('STEP 11: HYBRID MODEL EVALUATION')
print('='*65)

hybrid_y_probs = xgb_hybrid.predict_proba(hybrid_test_f)[:, 1]

prec_h, rec_h, thresh_h = precision_recall_curve(hybrid_y_test, hybrid_y_probs)
f2_h = np.where(
    (beta**2 * prec_h + rec_h) > 0,
    (1 + beta**2) * prec_h * rec_h / (beta**2 * prec_h + rec_h),
    0
)
best_thresh_h = thresh_h[np.argmax(f2_h[:-1])]
hybrid_y_pred = (hybrid_y_probs >= best_thresh_h).astype(int)
hybrid_auc    = roc_auc_score(hybrid_y_test, hybrid_y_probs)

print(f'Optimal F2 threshold : {best_thresh_h:.4f}')
print(f'Accuracy  : {accuracy_score(hybrid_y_test, hybrid_y_pred)*100:.2f}%')
print(f'Recall    : {recall_score(hybrid_y_test, hybrid_y_pred)*100:.2f}%')
print(f'F1 Score  : {f1_score(hybrid_y_test, hybrid_y_pred)*100:.2f}%')
print(f'ROC-AUC   : {hybrid_auc:.4f}')
print()
print(classification_report(hybrid_y_test, hybrid_y_pred, target_names=['No Purchase', 'Purchase']))

# Feature importance: GRU dims vs tabular
feat_imp_h    = pd.Series(xgb_hybrid.feature_importances_, index=hybrid_train.columns)
gru_imp_total = feat_imp_h[emb_cols].sum()
tab_imp_total = feat_imp_h[tab_features.columns].sum()
print(f'\nTotal importance — GRU dims : {gru_imp_total:.3f}  |  Tabular : {tab_imp_total:.3f}')
print(f'Top 5 GRU dims  : {feat_imp_h[emb_cols].nlargest(5).index.tolist()}')
print(f'Top 5 tabular   : {feat_imp_h[tab_features.columns].nlargest(5).index.tolist()}')

fpr_h, tpr_h, _ = roc_curve(hybrid_y_test, hybrid_y_probs)

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

cm_h = confusion_matrix(hybrid_y_test, hybrid_y_pred)
sns.heatmap(cm_h, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Purchase', 'Purchase'],
            yticklabels=['No Purchase', 'Purchase'], ax=axes[0])
axes[0].set_title('Hybrid — Confusion Matrix', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

feat_imp_h.nlargest(20).sort_values().plot(kind='barh', ax=axes[1], color='#55A868')
axes[1].set_title('Hybrid — Top 20 Feature Importances', fontweight='bold')
axes[1].set_xlabel('Importance Score')

axes[2].plot(fpr_h, tpr_h, color='#55A868', lw=2, label=f'Hybrid AUC = {hybrid_auc:.3f}')
axes[2].plot([0,1],[0,1],'k--',lw=1,label='Random')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('Hybrid — ROC-AUC', fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(GRU_OUTPUT_DIR, 'hybrid_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: hybrid_results.png')


# =============================================================================
# STEP 12 — Final 3-Model Comparison
# =============================================================================
print('\n' + '='*65)
print('STEP 12: FINAL COMPARISON — XGBoost vs GRU vs Hybrid')
print('='*65)

results = pd.DataFrame({
    'Model'    : ['XGBoost (tabular)', 'GRU (sequence)', 'Hybrid (GRU+XGB)'],
    'ROC-AUC'  : [xgb_auc,   gru_auc,   hybrid_auc],
    'Recall'   : [recall_score(xgb_y_test, xgb_y_pred),
                  recall_score(gru_y_test,    gru_y_pred),
                  recall_score(hybrid_y_test, hybrid_y_pred)],
    'F1 Score' : [f1_score(xgb_y_test, xgb_y_pred),
                  f1_score(gru_y_test,    gru_y_pred),
                  f1_score(hybrid_y_test, hybrid_y_pred)],
    'Accuracy' : [accuracy_score(xgb_y_test, xgb_y_pred),
                  accuracy_score(gru_y_test,    gru_y_pred),
                  accuracy_score(hybrid_y_test, hybrid_y_pred)],
    'Threshold': [0.5, float(best_thresh_g), float(best_thresh_h)]
}).set_index('Model').round(4)

print(results)

results.to_csv(os.path.join(GRU_OUTPUT_DIR, 'model_comparison.csv'))
print('Saved: model_comparison.csv')

thresholds_dict = {
    'xgboost_default'  : 0.5,
    'gru_f2_optimal'   : float(best_thresh_g),
    'hybrid_f2_optimal': float(best_thresh_h)
}
with open(os.path.join(GRU_OUTPUT_DIR, 'optimal_thresholds.json'), 'w') as f:
    json.dump(thresholds_dict, f, indent=2)
print('Saved: optimal_thresholds.json')

# ── BAR CHART COMPARISON ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
colors  = ['#4C72B0', '#DD8452', '#55A868']
metrics = ['ROC-AUC', 'Recall', 'F1 Score', 'Accuracy']

for ax, metric in zip(axes, metrics):
    bars = ax.bar(
        results.index, results[metric],
        color=colors, width=0.5, edgecolor='white', linewidth=1.2
    )
    ax.set_ylim(0, 1.12)
    ax.set_title(metric, fontweight='bold', fontsize=13)
    ax.set_xticklabels(results.index, rotation=18, ha='right', fontsize=9)
    ax.tick_params(axis='x', length=0)
    ax.spines[['top','right']].set_visible(False)
    for bar, val in zip(bars, results[metric]):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

plt.suptitle('Model Comparison — XGBoost vs GRU vs Hybrid',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(GRU_OUTPUT_DIR, 'model_comparison_bar.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── OVERLAID ROC CURVES ──────────────────────────────────────────────────────
plt.figure(figsize=(8, 7))
plt.plot(fpr_xgb, tpr_xgb, color='#4C72B0', lw=2.5, label=f'XGBoost   AUC = {xgb_auc:.3f}')
plt.plot(fpr_g,   tpr_g,   color='#DD8452', lw=2.5, label=f'GRU       AUC = {gru_auc:.3f}')
plt.plot(fpr_h,   tpr_h,   color='#55A868', lw=2.5, label=f'Hybrid    AUC = {hybrid_auc:.3f}')
plt.plot([0,1],[0,1],'k--',lw=1,label='Random baseline')
plt.fill_between(fpr_h, tpr_h, alpha=0.05, color='#55A868')
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate',  fontsize=13)
plt.title('ROC Curves — All Three Models', fontsize=14, fontweight='bold')
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRU_OUTPUT_DIR, 'roc_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: model_comparison_bar.png, roc_comparison.png')


# =============================================================================
# STEP 13 — GRU Psychology Test Cases
# =============================================================================
print('\n' + '='*65)
print('STEP 13: GRU PSYCHOLOGY TEST CASES')
print('='*65)

def secs(s): return np.log1p(s)

test_cases = {
    'VIP Focused Buyer'  : [[1,0,0.9,secs(0)],  [1,0,0.9,secs(12)],  [0,1,0.9,secs(8)]],
    'Window Shopper'     : [[1,0,0.1,secs(0)],  [1,0,0.15,secs(150)],[1,0,0.12,secs(300)],[1,0,0.2,secs(420)]],
    'Cart Abandoner'     : [[1,0,0.5,secs(0)],  [0,1,0.5,secs(25)],  [1,0,0.5,secs(1800)]],
    'Comparison Shopper' : [[1,0,0.3,secs(0)],  [1,0,0.7,secs(45)],  [1,0,0.5,secs(30)],
                            [1,0,0.6,secs(20)], [0,1,0.6,secs(15)]]
}
expected = {
    'VIP Focused Buyer'  : '> 70%',
    'Window Shopper'     : '< 15%',
    'Cart Abandoner'     : '20–50%',
    'Comparison Shopper' : '40–70%'
}

X_tc = pad_sequences(list(test_cases.values()), maxlen=MAX_STEPS, padding='post', dtype='float32')
preds_tc = gru_model.predict(X_tc, verbose=0)

print('TEST RESULTS:')
print('-' * 65)
for i, (name, _) in enumerate(test_cases.items()):
    prob = preds_tc[i][0] * 100
    print(f'{name:<25} : {prob:5.1f}%  (expected {expected[name]})')
print('-' * 65)


# =============================================================================
# STEP 14 — Final File Summary
# =============================================================================
print('\n' + '='*65)
print('ALL SAVED FILES')
print('='*65)
for fname in sorted(os.listdir(GRU_OUTPUT_DIR)):
    fpath = os.path.join(GRU_OUTPUT_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f'  {fname:<55} {size_kb:>8.1f} KB')

print('\n' + '='*65)
print('FINAL RESULTS SUMMARY')
print('='*65)
print(results)
print('\nPipeline complete. All outputs saved to ./' + GRU_OUTPUT_DIR + '/')