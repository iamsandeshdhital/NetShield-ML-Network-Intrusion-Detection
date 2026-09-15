"""Central configuration for the NetShield-ML pipeline.

Every path, constant and hyper-parameter used by more than one module lives
here so that a single edit changes the behaviour of the whole pipeline.

Authors: Anurag Jha, Sandesh Dhital
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"

for _directory in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DIR / "KDDTrain+.txt"
TEST_FILE = RAW_DIR / "KDDTest+.txt"

# Mirrors used by ``src.data_loader`` when the raw files are missing.
DATA_SOURCES = {
    TRAIN_FILE.name: [
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    ],
    TEST_FILE.name: [
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt",
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
    ],
}

# --------------------------------------------------------------------------
# Dataset schema
# --------------------------------------------------------------------------
# The NSL-KDD "+" files are headerless: 41 features, the attack label and a
# trailing difficulty score produced by the dataset authors.
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# Columns that are stored as integers but are semantically binary flags.  They
# are excluded from standardisation so their 0/1 meaning survives preprocessing.
BINARY_COLUMNS = ["land", "logged_in", "root_shell", "is_host_login", "is_guest_login"]

# Dropped before modelling: ``difficulty`` is dataset metadata, not a network
# observation, and leaking it would inflate every score.
METADATA_COLUMNS = ["difficulty"]

LABEL_COLUMN = "label"
TARGET_COLUMN = "is_attack"          # 0 = normal, 1 = attack
TARGET_NAMES = ["Normal", "Attack"]

# --------------------------------------------------------------------------
# Attack taxonomy (used for reporting, not for the binary target)
# --------------------------------------------------------------------------
ATTACK_CATEGORIES = {
    # Denial of Service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS",
    # Probe / surveillance
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # Remote to Local
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "worm": "R2L",
    # User to Root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R",
    "sqlattack": "U2R", "xterm": "U2R",
    # Benign traffic
    "normal": "Normal",
}

# --------------------------------------------------------------------------
# Pipeline hyper-parameters
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Feature selection
VARIANCE_THRESHOLD = 0.0      # drop features that never change
CORRELATION_THRESHOLD = 0.95  # drop one of any near-duplicate pair
N_SELECTED_FEATURES = 40      # kept by the mutual-information ranking
TOP_N_IMPORTANCES = 10        # size of the feature-importance chart

# --------------------------------------------------------------------------
# Plot styling
# --------------------------------------------------------------------------
FIGURE_DPI = 200
FIGURE_FORMAT = "png"
PALETTE = ["#2B6CB0", "#DD6B20", "#2F855A", "#805AD5", "#C53030"]
