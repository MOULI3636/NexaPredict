import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor, AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor, StackingClassifier, StackingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np
from tkinter import scrolledtext
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import shap
from pandastable import Table
import matplotlib.ticker as ticker
from matplotlib import rcParams
from sklearn.ensemble import IsolationForest
from scipy import stats

# Modern clean color scheme - Professional deep blue theme
BACKGROUND_COLOR = "#FFFFFF"
PRIMARY_COLOR = "#F8F9FA"
SECONDARY_COLOR = "#E9ECEF"
ACCENT_COLOR = "#2C3E50"  # Deep blue
HOVER_COLOR = "#34495E"   # Darker blue
TEXT_COLOR = "#2C3E50"
DANGER_COLOR = "#E74C3C"
SUCCESS_COLOR = "#27AE60"
INFO_COLOR = "#3498DB"
WARNING_COLOR = "#F39C12"
HIGHLIGHT_COLOR = "#2980B9"
BORDER_COLOR = "#2C3E50"

# Configure matplotlib style for white background
plt.style.use('default')
rcParams['axes.titlecolor'] = TEXT_COLOR
rcParams['axes.labelcolor'] = TEXT_COLOR
rcParams['xtick.color'] = TEXT_COLOR
rcParams['ytick.color'] = TEXT_COLOR
rcParams['figure.facecolor'] = BACKGROUND_COLOR
rcParams['axes.facecolor'] = BACKGROUND_COLOR
rcParams['grid.color'] = '#DDDDDD'

def load_file():
    """Function to load the file."""
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")])
    if file_path:
        try:
            global df, original_df
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            original_df = df.copy()  # Keep a copy of original data
            messagebox.showinfo("Success", "Dataset loaded successfully!")
            update_data_preview()
            enable_buttons()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

def enable_buttons():
    """Enable all analysis buttons after data is loaded."""
    prepare_data_btn.config(state=tk.NORMAL)
    visualize_btn.config(state=tk.NORMAL)
    analyze_dataset_btn.config(state=tk.NORMAL)
    corr_matrix_btn.config(state=tk.NORMAL)
    standardize_btn.config(state=tk.NORMAL)
    advanced_analysis_btn.config(state=tk.NORMAL)

def add_hover_effect(button, hover_color=HOVER_COLOR, original_color=None):
    """Function to add hover effect to a button."""
    if original_color is None:
        original_color = button.cget('bg')
    
    def on_enter(e):
        button.config(bg=hover_color)

    def on_leave(e):
        button.config(bg=original_color)

    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

def update_data_preview():
    """Update the data preview text widget with improved formatting."""
    data_preview.config(state=tk.NORMAL)
    data_preview.delete(1.0, tk.END)
    
    if df.empty:
        data_preview.insert(tk.END, "No data loaded. Please click 'Load Dataset' to begin.")
    else:
        # Header with dataset info
        data_preview.insert(tk.END, "DATASET OVERVIEW\n", "header")
        data_preview.insert(tk.END, f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n", "info")
        
        # Dataset preview
        data_preview.insert(tk.END, "PREVIEW (First 5 rows)\n", "header")
        data_preview.insert(tk.END, df.head().to_string() + "\n\n", "data")
        
        # Dataset information
        data_preview.insert(tk.END, "DATA TYPES\n", "header")
        data_types = df.dtypes.to_string()
        data_preview.insert(tk.END, data_types + "\n\n", "data")
        
        # Null value counts
        data_preview.insert(tk.END, "MISSING VALUES\n", "header")
        null_counts = df.isnull().sum()
        if null_counts.sum() == 0:
            data_preview.insert(tk.END, "No missing values found in the dataset.\n", "info")
        else:
            for col, count in null_counts.items():
                if count > 0:
                    data_preview.insert(tk.END, f"{col}: {count} null values ({count/len(df):.1%})\n", "warning")
    
    data_preview.config(state=tk.DISABLED)

def show_workflow():
    """Show the workflow of the dashboard."""
    workflow_window = tk.Toplevel(root)
    workflow_window.title("Dashboard Workflow")
    workflow_window.geometry("800x600")
    workflow_window.configure(bg=BACKGROUND_COLOR)
    
    # Position the window
    workflow_window.geometry(f"+{root.winfo_x()+100}+{root.winfo_y()+100}")

    # Main container with stylish border
    main_frame = tk.Frame(workflow_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Title
    title_label = tk.Label(main_frame, text="Dashboard Workflow Guide", 
                         font=("Helvetica", 18, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    title_label.pack(pady=(10, 20))

    # Workflow steps
    workflow_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=20, 
                                            bg=SECONDARY_COLOR, fg=TEXT_COLOR, 
                                            font=("Helvetica", 11), padx=15, pady=15)
    workflow_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    workflow_steps = """
ML DATA ANALYZER - WORKFLOW GUIDE

1. LOAD DATASET
   -> Load your CSV or Excel file
   -> Dataset overview will be displayed

2. PREPARE DATA
   -> Handle missing values (drop, mean, median, mode, zero)
   -> Feature selection (choose input features and target)
   -> Data type conversion (numeric to categorical and vice versa)
   -> Outlier detection and removal
   -> PCA for dimensionality reduction
   -> Statistical analysis

3. STANDARDIZE DATA
   -> Apply StandardScaler to normalize numeric features
   -> View standardized dataset

4. ANALYZE DATASET
   -> Automatic dataset analysis and problem type detection
   -> Choose between supervised and unsupervised learning
   -> Model comparison and evaluation

5. VISUALIZE DATA
   -> Various plot types: Pie Chart, Bar Chart, Count Plot, Scatter Plot, Histogram, 
      Dist Plot, KDE Plot, Line Plot, Violin Plot, Area Plot, Box Plot, 3D Plot

6. CORRELATION MATRIX
   -> View feature correlations with heatmap

7. ADVANCED ANALYSIS
   -> SHAP analysis for model interpretability
   -> Ensemble method comparisons
   -> Feature importance visualization

8. MODEL TRAINING & EVALUATION
   -> Multiple algorithms for classification, regression, and clustering
   -> Performance metrics and visualization
   -> Model comparison charts

TIPS:
- Start with data preparation for better results
- Use visualization to understand data patterns
- Compare multiple models to find the best performer
- Use SHAP analysis to understand model decisions
"""
    
    workflow_text.insert(tk.END, workflow_steps)
    workflow_text.config(state=tk.DISABLED)

    # Close button
    close_btn = tk.Button(main_frame, text="Close", command=workflow_window.destroy,
                         bg=DANGER_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground="#B71C1C")
    close_btn.pack(pady=(10, 20))
    add_hover_effect(close_btn, hover_color="#B71C1C", original_color=DANGER_COLOR)

def show_correlation_matrix():
    """Display correlation matrix in a separate window with improved styling."""
    if df.empty:
        messagebox.showerror("Error", "No data loaded!")
        return
    
    try:
        corr_window = tk.Toplevel(root)
        corr_window.title("Correlation Matrix")
        corr_window.geometry("1100x900")
        corr_window.configure(bg=BACKGROUND_COLOR)
        
        # Position the window
        corr_window.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

        # Main container with stylish border
        main_frame = tk.Frame(corr_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(main_frame, text="Feature Correlation Matrix", 
                             font=("Helvetica", 18, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
        title_label.pack(pady=(10, 20))

        # Calculate correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            messagebox.showerror("Error", "No numeric columns to calculate correlation!")
            corr_window.destroy()
            return
            
        corr_matrix = numeric_df.corr()
        
        # Create figure with improved styling
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap with better styling
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                   annot_kws={"size": 10, "color": "black"}, fmt=".2f", 
                   linewidths=.5, cbar_kws={"shrink": 0.8})
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Add border to the plot
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color(BORDER_COLOR)
            spine.set_linewidth(2)

        # Embed in Tkinter with stylish frame
        plot_frame = tk.Frame(main_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, highlightbackground=BORDER_COLOR, highlightthickness=1)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Close button with modern style
        close_btn = tk.Button(main_frame, text="Close", command=corr_window.destroy,
                            bg=ACCENT_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                            relief=tk.FLAT, padx=30, pady=8, bd=0, highlightthickness=0,
                            activebackground=HOVER_COLOR)
        close_btn.pack(pady=(10, 20))
        add_hover_effect(close_btn)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create correlation matrix: {e}")

def analyze_dataset():
    """Analyze dataset and suggest appropriate analysis type."""
    if df.empty:
        messagebox.showerror("Error", "No data loaded!")
        return
    
    try:
        # Determine dataset characteristics
        n_rows, n_cols = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Check if it's a supervised learning problem (has target variable)
        has_target = len(df.columns) > 1  # Assuming last column is target
        
        # Determine problem type
        problem_type = "Unknown"
        suggestion = ""
        
        if has_target:
            target_col = df.columns[-1]
            if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() < 10:
                problem_type = "Classification"
                suggestion = "This appears to be a classification problem. The target variable is categorical."
            else:
                problem_type = "Regression"
                suggestion = "This appears to be a regression problem. The target variable is continuous."
        else:
            problem_type = "Unsupervised"
            suggestion = "This appears to be an unsupervised learning problem. No clear target variable found."
        
        # Create analysis popup
        popup = tk.Toplevel(root)
        popup.title("Dataset Analysis Results")
        popup.geometry("800x600")
        popup.configure(bg=BACKGROUND_COLOR)
        
        # Position the popup
        popup.geometry(f"+{root.winfo_x()+root.winfo_width()//2-400}+{root.winfo_y()+root.winfo_height()//2-300}")

        # Main container with stylish border
        main_frame = tk.Frame(popup, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        title_label = tk.Label(main_frame, text="Dataset Analysis", 
                             font=("Helvetica", 18, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
        title_label.pack(pady=(10, 20))

        # Analysis results frame
        results_frame = tk.Frame(main_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, highlightbackground=BORDER_COLOR, highlightthickness=1)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        result_text = tk.Text(results_frame, wrap=tk.WORD, height=15, bg=SECONDARY_COLOR, 
                            fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Helvetica", 11),
                            padx=10, pady=10)
        scrollbar = ttk.Scrollbar(results_frame, command=result_text.yview)
        result_text.config(yscrollcommand=scrollbar.set)
        
        result_text.insert(tk.END, "DATASET CHARACTERISTICS:\n\n", "header")
        result_text.insert(tk.END, f"• Dimensions: {n_rows} rows, {n_cols} columns\n")
        result_text.insert(tk.END, f"• Numeric columns: {len(numeric_cols)}\n")
        result_text.insert(tk.END, f"• Categorical columns: {len(categorical_cols)}\n")
        result_text.insert(tk.END, f"• Missing values: {df.isnull().sum().sum()}\n\n")
        
        result_text.insert(tk.END, "ANALYSIS RESULTS:\n\n", "header")
        result_text.insert(tk.END, f"• Problem Type: {problem_type}\n")
        result_text.insert(tk.END, f"• Suggestion: {suggestion}\n\n")
        
        if has_target:
            target_col = df.columns[-1]
            result_text.insert(tk.END, f"• Target Variable: {target_col}\n")
            if problem_type == "Classification":
                result_text.insert(tk.END, f"• Number of classes: {df[target_col].nunique()}\n")
                result_text.insert(tk.END, f"• Class distribution:\n")
                class_counts = df[target_col].value_counts()
                for cls, count in class_counts.items():
                    result_text.insert(tk.END, f"  - {cls}: {count} samples ({count/len(df)*100:.1f}%)\n")
            else:
                result_text.insert(tk.END, f"• Target statistics:\n")
                result_text.insert(tk.END, f"  - Min: {df[target_col].min():.2f}\n")
                result_text.insert(tk.END, f"  - Max: {df[target_col].max():.2f}\n")
                result_text.insert(tk.END, f"  - Mean: {df[target_col].mean():.2f}\n")
                result_text.insert(tk.END, f"  - Std: {df[target_col].std():.2f}\n")
        
        result_text.config(state=tk.DISABLED)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Button frame
        button_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
        button_frame.pack(pady=20)

        if problem_type in ["Classification", "Regression"]:
            supervised_btn = tk.Button(button_frame, text="Supervised Learning", 
                                    command=lambda: [popup.destroy(), show_supervised_options(problem_type)],
                                    bg=SUCCESS_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                                    relief=tk.FLAT, padx=20, pady=10, bd=0, highlightthickness=0,
                                    activebackground=HOVER_COLOR)
            supervised_btn.pack(side=tk.LEFT, padx=10)
            add_hover_effect(supervised_btn, hover_color=HOVER_COLOR, original_color=SUCCESS_COLOR)

        unsupervised_btn = tk.Button(button_frame, text="Unsupervised Learning", 
                                   command=lambda: [popup.destroy(), show_unsupervised_options()],
                                   bg=INFO_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                                   relief=tk.FLAT, padx=20, pady=10, bd=0, highlightthickness=0,
                                   activebackground=HOVER_COLOR)
        unsupervised_btn.pack(side=tk.LEFT, padx=10)
        add_hover_effect(unsupervised_btn, hover_color=HOVER_COLOR, original_color=INFO_COLOR)

        # Close button
        close_btn = tk.Button(main_frame, text="Close", command=popup.destroy,
                            bg=DANGER_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                            relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                            activebackground="#B71C1C")
        close_btn.pack(pady=(10, 20))
        add_hover_effect(close_btn, hover_color="#B71C1C", original_color=DANGER_COLOR)

    except Exception as e:
        messagebox.showerror("Error", f"Error during dataset analysis: {e}")

def show_supervised_options(problem_type):
    """Show supervised learning options based on problem type."""
    popup = tk.Toplevel(root)
    popup.title(f"Supervised Learning - {problem_type}")
    popup.geometry("700x600")
    popup.configure(bg=BACKGROUND_COLOR)
    
    # Position the popup
    popup.geometry(f"+{root.winfo_x()+root.winfo_width()//2-350}+{root.winfo_y()+root.winfo_height()//2-300}")

    # Main container with stylish border
    main_frame = tk.Frame(popup, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    title_label = tk.Label(main_frame, text=f"Supervised Learning - {problem_type}", 
                         font=("Helvetica", 18, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    title_label.pack(pady=(10, 20))

    # Algorithms frame
    algorithms_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    algorithms_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # Create notebook for different algorithm categories
    notebook = ttk.Notebook(algorithms_frame)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Linear Models Tab
    linear_tab = ttk.Frame(notebook)
    notebook.add(linear_tab, text="Linear Models")
    setup_linear_models_tab(linear_tab, problem_type)

    # Tree-based Models Tab
    tree_tab = ttk.Frame(notebook)
    notebook.add(tree_tab, text="Tree-based Models")
    setup_tree_models_tab(tree_tab, problem_type)

    # Ensemble Models Tab
    ensemble_tab = ttk.Frame(notebook)
    notebook.add(ensemble_tab, text="Ensemble Models")
    setup_ensemble_models_tab(ensemble_tab, problem_type)

    # Other Models Tab
    other_tab = ttk.Frame(notebook)
    notebook.add(other_tab, text="Other Models")
    setup_other_models_tab(other_tab, problem_type)

    # Compare Models button
    compare_btn = tk.Button(main_frame, text="Compare All Models", 
                          command=lambda: [popup.destroy(), compare_supervised_models(problem_type)],
                          bg=WARNING_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                          relief=tk.FLAT, padx=20, pady=10, bd=0, highlightthickness=0,
                          activebackground=HOVER_COLOR)
    compare_btn.pack(pady=(10, 20))
    add_hover_effect(compare_btn, hover_color=HOVER_COLOR, original_color=WARNING_COLOR)

    # Close button
    close_btn = tk.Button(main_frame, text="Close", command=popup.destroy,
                         bg=DANGER_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground="#B71C1C")
    close_btn.pack(pady=(10, 20))
    add_hover_effect(close_btn, hover_color="#B71C1C", original_color=DANGER_COLOR)

def setup_linear_models_tab(tab, problem_type):
    """Setup linear models tab."""
    tab.configure(style='TFrame')
    
    # Linear models
    linear_models = []
    if problem_type == "Classification":
        linear_models = [
            ("Logistic Regression", run_logistic_regression),
            ("Linear Discriminant Analysis", run_lda),
            ("Quadratic Discriminant Analysis", run_qda)
        ]
    else:
        linear_models = [
            ("Linear Regression", run_linear_regression),
            ("Ridge Regression", run_ridge_regression),
            ("Lasso Regression", run_lasso_regression),
            ("Elastic Net", run_elastic_net)
        ]
    
    for model_name, model_func in linear_models:
        btn = tk.Button(tab, text=model_name, command=model_func,
                       bg=ACCENT_COLOR, fg="white", font=("Helvetica", 10, "bold"),
                       relief=tk.FLAT, padx=20, pady=10, bd=0, highlightthickness=0,
                       activebackground=HOVER_COLOR, width=20)
        btn.pack(pady=5)
        add_hover_effect(btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

def setup_tree_models_tab(tab, problem_type):
    """Setup tree-based models tab."""
    tab.configure(style='TFrame')
    
    # Tree-based models
    tree_models = []
    if problem_type == "Classification":
        tree_models = [
            ("Decision Tree Classifier", run_decision_tree_classifier),
            ("Random Forest Classifier", run_random_forest_classifier),
            ("Extra Trees Classifier", run_extra_trees_classifier)
        ]
    else:
        tree_models = [
            ("Decision Tree Regressor", run_decision_tree_regressor),
            ("Random Forest Regressor", run_random_forest_regressor),
            ("Extra Trees Regressor", run_extra_trees_regressor)
        ]
    
    for model_name, model_func in tree_models:
        btn = tk.Button(tab, text=model_name, command=model_func,
                       bg=ACCENT_COLOR, fg="white", font=("Helvetica", 10, "bold"),
                       relief=tk.FLAT, padx=20, pady=10, bd=0, highlightthickness=0,
                       activebackground=HOVER_COLOR, width=20)
        btn.pack(pady=5)
        add_hover_effect(btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

def setup_ensemble_models_tab(tab, problem_type):
    """Setup ensemble models tab."""
    tab.configure(style='TFrame')
    
    # Ensemble models
    ensemble_models = []
    if problem_type == "Classification":
        ensemble_models = [
            ("Gradient Boosting Classifier", run_gradient_boosting_classifier),
            ("XGBoost Classifier", run_xgboost_classifier),
            ("LightGBM Classifier", run_lightgbm_classifier),
            ("CatBoost Classifier", run_catboost_classifier),
            ("AdaBoost Classifier", run_adaboost_classifier),
            ("Bagging Classifier", run_bagging_classifier)
        ]
    else:
        ensemble_models = [
            ("Gradient Boosting Regressor", run_gradient_boosting_regressor),
            ("XGBoost Regressor", run_xgboost_regressor),
            ("LightGBM Regressor", run_lightgbm_regressor),
            ("CatBoost Regressor", run_catboost_regressor),
            ("AdaBoost Regressor", run_adaboost_regressor),
            ("Bagging Regressor", run_bagging_regressor)
        ]
    
    for model_name, model_func in ensemble_models:
        btn = tk.Button(tab, text=model_name, command=model_func,
                       bg=ACCENT_COLOR, fg="white", font=("Helvetica", 10, "bold"),
                       relief=tk.FLAT, padx=20, pady=10, bd=0, highlightthickness=0,
                       activebackground=HOVER_COLOR, width=20)
        btn.pack(pady=5)
        add_hover_effect(btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

def setup_other_models_tab(tab, problem_type):
    """Setup other models tab."""
    tab.configure(style='TFrame')
    
    # Other models
    other_models = []
    if problem_type == "Classification":
        other_models = [
            ("K-Nearest Neighbors", run_knn_classifier),
            ("Gaussian Naive Bayes", run_gaussian_nb),
            ("Multinomial Naive Bayes", run_multinomial_nb),
            ("Bernoulli Naive Bayes", run_bernoulli_nb),
            ("Support Vector Machine", run_svm_classifier)
        ]
    else:
        other_models = [
            ("K-Nearest Neighbors Regressor", run_knn_regressor),
            ("Support Vector Regressor", run_svm_regressor)
        ]
    
    for model_name, model_func in other_models:
        btn = tk.Button(tab, text=model_name, command=model_func,
                       bg=ACCENT_COLOR, fg="white", font=("Helvetica", 10, "bold"),
                       relief=tk.FLAT, padx=20, pady=10, bd=0, highlightthickness=0,
                       activebackground=HOVER_COLOR, width=20)
        btn.pack(pady=5)
        add_hover_effect(btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

def show_unsupervised_options():
    """Show unsupervised learning options."""
    popup = tk.Toplevel(root)
    popup.title("Unsupervised Learning")
    popup.geometry("600x500")
    popup.configure(bg=BACKGROUND_COLOR)
    
    # Position the popup
    popup.geometry(f"+{root.winfo_x()+root.winfo_width()//2-300}+{root.winfo_y()+root.winfo_height()//2-250}")

    # Main container with stylish border
    main_frame = tk.Frame(popup, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    title_label = tk.Label(main_frame, text="Unsupervised Learning Algorithms", 
                         font=("Helvetica", 18, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    title_label.pack(pady=(10, 20))

    # Clustering algorithms
    clustering_algorithms = [
        ("K-Means Clustering", run_kmeans),
        ("DBSCAN Clustering", run_dbscan),
        ("Agglomerative Clustering", run_agglomerative),
        ("Gaussian Mixture Models", run_gaussian_mixture)
    ]
    
    for algo_name, algo_func in clustering_algorithms:
        btn = tk.Button(main_frame, text=algo_name, command=algo_func,
                       bg=ACCENT_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                       relief=tk.FLAT, padx=20, pady=12, bd=0, highlightthickness=0,
                       activebackground=HOVER_COLOR, width=25)
        btn.pack(pady=8)
        add_hover_effect(btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

    # Compare Models button
    compare_btn = tk.Button(main_frame, text="Compare Clustering Models", 
                          command=lambda: [popup.destroy(), compare_unsupervised_models()],
                          bg=WARNING_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                          relief=tk.FLAT, padx=20, pady=10, bd=0, highlightthickness=0,
                          activebackground=HOVER_COLOR)
    compare_btn.pack(pady=(20, 10))
    add_hover_effect(compare_btn, hover_color=HOVER_COLOR, original_color=WARNING_COLOR)

    # Close button
    close_btn = tk.Button(main_frame, text="Close", command=popup.destroy,
                         bg=DANGER_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground="#B71C1C")
    close_btn.pack(pady=(10, 20))
    add_hover_effect(close_btn, hover_color="#B71C1C", original_color=DANGER_COLOR)

# =============================================================================
# SUPERVISED LEARNING ALGORITHMS IMPLEMENTATION
# =============================================================================

def run_logistic_regression():
    """Run Logistic Regression with proper visualization."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Display results
        show_classification_results("Logistic Regression", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Logistic Regression: {e}")

def run_linear_regression():
    """Run Linear Regression with proper visualization."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display results
        show_regression_results("Linear Regression", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Linear Regression: {e}")

def run_ridge_regression():
    """Run Ridge Regression."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("Ridge Regression", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Ridge Regression: {e}")

def run_lasso_regression():
    """Run Lasso Regression."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = Lasso(alpha=0.1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("Lasso Regression", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Lasso Regression: {e}")

def run_elastic_net():
    """Run Elastic Net Regression."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("Elastic Net", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Elastic Net: {e}")

def run_lda():
    """Run Linear Discriminant Analysis."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearDiscriminantAnalysis()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("Linear Discriminant Analysis", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in LDA: {e}")

def run_qda():
    """Run Quadratic Discriminant Analysis."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = QuadraticDiscriminantAnalysis()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("Quadratic Discriminant Analysis", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in QDA: {e}")

def run_gaussian_nb():
    """Run Gaussian Naive Bayes."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("Gaussian Naive Bayes", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Gaussian Naive Bayes: {e}")

def run_multinomial_nb():
    """Run Multinomial Naive Bayes."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("Multinomial Naive Bayes", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Multinomial Naive Bayes: {e}")

def run_bernoulli_nb():
    """Run Bernoulli Naive Bayes."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = BernoulliNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("Bernoulli Naive Bayes", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Bernoulli Naive Bayes: {e}")

def run_knn_classifier():
    """Run K-Nearest Neighbors Classifier."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("K-Nearest Neighbors", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in K-Nearest Neighbors: {e}")

def run_knn_regressor():
    """Run K-Nearest Neighbors Regressor."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("K-Nearest Neighbors Regressor", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in K-Nearest Neighbors Regressor: {e}")

def run_decision_tree_classifier():
    """Run Decision Tree Classifier."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("Decision Tree Classifier", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Decision Tree Classifier: {e}")

def run_decision_tree_regressor():
    """Run Decision Tree Regressor."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("Decision Tree Regressor", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Decision Tree Regressor: {e}")

def run_random_forest_classifier():
    """Run Random Forest Classifier."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("Random Forest Classifier", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Random Forest Classifier: {e}")

def run_random_forest_regressor():
    """Run Random Forest Regressor."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("Random Forest Regressor", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Random Forest Regressor: {e}")

def run_extra_trees_classifier():
    """Run Extra Trees Classifier."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = ExtraTreesClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("Extra Trees Classifier", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Extra Trees Classifier: {e}")

def run_extra_trees_regressor():
    """Run Extra Trees Regressor."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = ExtraTreesRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("Extra Trees Regressor", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Extra Trees Regressor: {e}")

def run_gradient_boosting_classifier():
    """Run Gradient Boosting Classifier."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("Gradient Boosting Classifier", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Gradient Boosting Classifier: {e}")

def run_gradient_boosting_regressor():
    """Run Gradient Boosting Regressor."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("Gradient Boosting Regressor", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Gradient Boosting Regressor: {e}")

def run_xgboost_classifier():
    """Run XGBoost Classifier."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = XGBClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("XGBoost Classifier", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in XGBoost Classifier: {e}")

def run_xgboost_regressor():
    """Run XGBoost Regressor."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("XGBoost Regressor", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in XGBoost Regressor: {e}")

def run_lightgbm_classifier():
    """Run LightGBM Classifier."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LGBMClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("LightGBM Classifier", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in LightGBM Classifier: {e}")

def run_lightgbm_regressor():
    """Run LightGBM Regressor."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LGBMRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("LightGBM Regressor", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in LightGBM Regressor: {e}")

def run_catboost_classifier():
    """Run CatBoost Classifier."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = CatBoostClassifier(iterations=100, random_state=42, verbose=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("CatBoost Classifier", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in CatBoost Classifier: {e}")

def run_catboost_regressor():
    """Run CatBoost Regressor."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = CatBoostRegressor(iterations=100, random_state=42, verbose=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("CatBoost Regressor", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in CatBoost Regressor: {e}")

def run_adaboost_classifier():
    """Run AdaBoost Classifier."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = AdaBoostClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("AdaBoost Classifier", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in AdaBoost Classifier: {e}")

def run_adaboost_regressor():
    """Run AdaBoost Regressor."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = AdaBoostRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("AdaBoost Regressor", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in AdaBoost Regressor: {e}")

def run_bagging_classifier():
    """Run Bagging Classifier."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = BaggingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("Bagging Classifier", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Bagging Classifier: {e}")

def run_bagging_regressor():
    """Run Bagging Regressor."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = BaggingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("Bagging Regressor", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Bagging Regressor: {e}")

def run_svm_classifier():
    """Run Support Vector Machine Classifier."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        show_classification_results("Support Vector Machine", model, X_test, y_test, y_pred, accuracy, cm, report)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Support Vector Machine: {e}")

def run_svm_regressor():
    """Run Support Vector Regressor."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        show_regression_results("Support Vector Regressor", model, X_test, y_test, y_pred, mse, r2)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Support Vector Regressor: {e}")

# =============================================================================
# UNSUPERVISED LEARNING ALGORITHMS IMPLEMENTATION
# =============================================================================

def run_kmeans():
    """Run K-Means Clustering."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.select_dtypes(include=[np.number])
        
        if X.empty:
            messagebox.showerror("Error", "No numeric columns found for clustering!")
            return
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal k using elbow method
        inertias = []
        k_range = range(2, min(11, X.shape[0] + 1))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Use optimal k (simplified - in practice, use elbow point)
        optimal_k = 3  # Simplified for demo
        
        # Fit with optimal k
        model = KMeans(n_clusters=optimal_k, random_state=42)
        labels = model.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, labels)
        
        show_clustering_results("K-Means Clustering", model, X_scaled, labels, silhouette_avg, inertias, k_range)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in K-Means Clustering: {e}")

def run_dbscan():
    """Run DBSCAN Clustering."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.select_dtypes(include=[np.number])
        
        if X.empty:
            messagebox.showerror("Error", "No numeric columns found for clustering!")
            return
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(X_scaled)
        
        # Calculate silhouette score (if more than 1 cluster)
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            silhouette_avg = silhouette_score(X_scaled, labels)
        else:
            silhouette_avg = -1
        
        show_clustering_results("DBSCAN Clustering", model, X_scaled, labels, silhouette_avg)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in DBSCAN Clustering: {e}")

def run_agglomerative():
    """Run Agglomerative Clustering."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.select_dtypes(include=[np.number])
        
        if X.empty:
            messagebox.showerror("Error", "No numeric columns found for clustering!")
            return
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = AgglomerativeClustering(n_clusters=3)
        labels = model.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, labels)
        
        show_clustering_results("Agglomerative Clustering", model, X_scaled, labels, silhouette_avg)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Agglomerative Clustering: {e}")

def run_gaussian_mixture():
    """Run Gaussian Mixture Models."""
    try:
        from sklearn.mixture import GaussianMixture
        
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.select_dtypes(include=[np.number])
        
        if X.empty:
            messagebox.showerror("Error", "No numeric columns found for clustering!")
            return
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = GaussianMixture(n_components=3, random_state=42)
        labels = model.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, labels)
        
        show_clustering_results("Gaussian Mixture Models", model, X_scaled, labels, silhouette_avg)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in Gaussian Mixture Models: {e}")

# =============================================================================
# RESULTS DISPLAY FUNCTIONS
# =============================================================================

def show_classification_results(model_name, model, X_test, y_test, y_pred, accuracy, cm, report):
    """Display classification results."""
    results_window = tk.Toplevel(root)
    results_window.title(f"{model_name} Results")
    results_window.geometry("1000x700")
    results_window.configure(bg=BACKGROUND_COLOR)
    
    # Position the window
    results_window.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

    # Main container with stylish border
    main_frame = tk.Frame(results_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    title_label = tk.Label(main_frame, text=f"{model_name} Results", 
                         font=("Helvetica", 18, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    title_label.pack(pady=(10, 20))

    # Results frame
    results_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # Text results
    text_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, highlightbackground=BORDER_COLOR, highlightthickness=1)
    text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

    result_text = tk.Text(text_frame, wrap=tk.WORD, height=15, bg=SECONDARY_COLOR, 
                        fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Courier", 10),
                        padx=10, pady=10)
    scrollbar = ttk.Scrollbar(text_frame, command=result_text.yview)
    result_text.config(yscrollcommand=scrollbar.set)
    
    result_text.insert(tk.END, f"MODEL: {model_name}\n\n")
    result_text.insert(tk.END, f"Accuracy: {accuracy:.4f}\n\n")
    result_text.insert(tk.END, "Classification Report:\n")
    result_text.insert(tk.END, f"{report}\n")
    result_text.insert(tk.END, "Confusion Matrix:\n")
    result_text.insert(tk.END, f"{cm}\n")
    
    result_text.config(state=tk.DISABLED)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Confusion matrix plot
    plot_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, highlightbackground=BORDER_COLOR, highlightthickness=1)
    plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted Labels", fontsize=12)
    ax.set_ylabel("True Labels", fontsize=12)
    
    # Add border to the plot
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(2)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Close button
    close_btn = tk.Button(main_frame, text="Close", command=results_window.destroy,
                         bg=DANGER_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground="#B71C1C")
    close_btn.pack(pady=(10, 20))
    add_hover_effect(close_btn, hover_color="#B71C1C", original_color=DANGER_COLOR)

def show_regression_results(model_name, model, X_test, y_test, y_pred, mse, r2):
    """Display regression results."""
    results_window = tk.Toplevel(root)
    results_window.title(f"{model_name} Results")
    results_window.geometry("1000x700")
    results_window.configure(bg=BACKGROUND_COLOR)
    
    # Position the window
    results_window.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

    # Main container with stylish border
    main_frame = tk.Frame(results_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    title_label = tk.Label(main_frame, text=f"{model_name} Results", 
                         font=("Helvetica", 18, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    title_label.pack(pady=(10, 20))

    # Results frame
    results_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # Text results
    text_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, highlightbackground=BORDER_COLOR, highlightthickness=1)
    text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

    result_text = tk.Text(text_frame, wrap=tk.WORD, height=15, bg=SECONDARY_COLOR, 
                        fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Courier", 10),
                        padx=10, pady=10)
    scrollbar = ttk.Scrollbar(text_frame, command=result_text.yview)
    result_text.config(yscrollcommand=scrollbar.set)
    
    result_text.insert(tk.END, f"MODEL: {model_name}\n\n")
    result_text.insert(tk.END, f"Mean Squared Error: {mse:.4f}\n")
    result_text.insert(tk.END, f"R² Score: {r2:.4f}\n\n")
    
    if hasattr(model, 'coef_'):
        result_text.insert(tk.END, "Feature Coefficients:\n")
        if len(model.coef_.shape) == 1:
            for i, coef in enumerate(model.coef_):
                result_text.insert(tk.END, f"  Feature {i+1}: {coef:.4f}\n")
        else:
            for i, coef_arr in enumerate(model.coef_):
                result_text.insert(tk.END, f"  Class {i}: {coef_arr}\n")
    
    if hasattr(model, 'intercept_'):
        result_text.insert(tk.END, f"Intercept: {model.intercept_:.4f}\n")
    
    result_text.config(state=tk.DISABLED)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # True vs Predicted plot
    plot_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, highlightbackground=BORDER_COLOR, highlightthickness=1)
    plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.7, color=ACCENT_COLOR)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("True Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.set_title("True vs Predicted Values", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add border to the plot
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(2)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Close button
    close_btn = tk.Button(main_frame, text="Close", command=results_window.destroy,
                         bg=DANGER_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground="#B71C1C")
    close_btn.pack(pady=(10, 20))
    add_hover_effect(close_btn, hover_color="#B71C1C", original_color=DANGER_COLOR)

def show_clustering_results(model_name, model, X, labels, silhouette_score, inertias=None, k_range=None):
    """Display clustering results."""
    results_window = tk.Toplevel(root)
    results_window.title(f"{model_name} Results")
    results_window.geometry("1000x700")
    results_window.configure(bg=BACKGROUND_COLOR)
    
    # Position the window
    results_window.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

    # Main container with stylish border
    main_frame = tk.Frame(results_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    title_label = tk.Label(main_frame, text=f"{model_name} Results", 
                         font=("Helvetica", 18, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    title_label.pack(pady=(10, 20))

    # Results frame
    results_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # Text results
    text_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, highlightbackground=BORDER_COLOR, highlightthickness=1)
    text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

    result_text = tk.Text(text_frame, wrap=tk.WORD, height=15, bg=SECONDARY_COLOR, 
                        fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Courier", 10),
                        padx=10, pady=10)
    scrollbar = ttk.Scrollbar(text_frame, command=result_text.yview)
    result_text.config(yscrollcommand=scrollbar.set)
    
    result_text.insert(tk.END, f"MODEL: {model_name}\n\n")
    result_text.insert(tk.END, f"Number of clusters: {len(set(labels))}\n")
    result_text.insert(tk.END, f"Silhouette Score: {silhouette_score:.4f}\n\n")
    result_text.insert(tk.END, "Cluster distribution:\n")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        result_text.insert(tk.END, f"  Cluster {cluster}: {count} samples\n")
    
    result_text.config(state=tk.DISABLED)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Clustering plot
    plot_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, highlightbackground=BORDER_COLOR, highlightthickness=1)
    plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    
    if X.shape[1] >= 2:
        # 2D scatter plot
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
        ax.set_xlabel("Feature 1", fontsize=12)
        ax.set_ylabel("Feature 2", fontsize=12)
        ax.set_title("Cluster Visualization", fontsize=14)
        plt.colorbar(scatter, ax=ax)
    elif inertias is not None and k_range is not None:
        # Elbow plot
        ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel("Number of Clusters (k)", fontsize=12)
        ax.set_ylabel("Inertia", fontsize=12)
        ax.set_title("Elbow Method", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add border to the plot
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(2)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Close button
    close_btn = tk.Button(main_frame, text="Close", command=results_window.destroy,
                         bg=DANGER_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground="#B71C1C")
    close_btn.pack(pady=(10, 20))
    add_hover_effect(close_btn, hover_color="#B71C1C", original_color=DANGER_COLOR)

# =============================================================================
# MODEL COMPARISON FUNCTIONS
# =============================================================================

def compare_supervised_models(problem_type):
    """Compare all supervised learning models."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Get data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target for classification
        if problem_type == "Classification" and y.dtype == 'object':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Define models based on problem type
        if problem_type == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(max_depth=5),
                "Random Forest": RandomForestClassifier(n_estimators=100),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
                "XGBoost": XGBClassifier(n_estimators=100),
                "LightGBM": LGBMClassifier(n_estimators=100),
                "CatBoost": CatBoostClassifier(iterations=100, verbose=False),
                "SVM": SVC()
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Elastic Net": ElasticNet(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(max_depth=5),
                "Random Forest": RandomForestRegressor(n_estimators=100),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100),
                "XGBoost": XGBRegressor(n_estimators=100),
                "LightGBM": LGBMRegressor(n_estimators=100),
                "CatBoost": CatBoostRegressor(iterations=100, verbose=False),
                "SVM": SVR()
            }
        
        # Train and evaluate models
        results = []
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if problem_type == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    results.append({"Model": name, "Accuracy": accuracy})
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    results.append({"Model": name, "MSE": mse, "R²": r2})
                    
            except Exception as e:
                print(f"Error with {name}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Display comparison results
        show_model_comparison(results_df, problem_type)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in model comparison: {e}")

def compare_unsupervised_models():
    """Compare all unsupervised learning models."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        X = df.select_dtypes(include=[np.number])
        
        if X.empty:
            messagebox.showerror("Error", "No numeric columns found for clustering!")
            return
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define models
        models = {
            "K-Means": KMeans(n_clusters=3, random_state=42),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
            "Agglomerative": AgglomerativeClustering(n_clusters=3)
        }
        
        # Train and evaluate models
        results = []
        for name, model in models.items():
            try:
                labels = model.fit_predict(X_scaled)
                unique_labels = set(labels)
                
                if len(unique_labels) > 1:
                    silhouette = silhouette_score(X_scaled, labels)
                else:
                    silhouette = -1
                
                results.append({
                    "Model": name, 
                    "Silhouette Score": silhouette,
                    "Clusters": len(unique_labels)
                })
                
            except Exception as e:
                print(f"Error with {name}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Display comparison results
        show_model_comparison(results_df, "Clustering")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in model comparison: {e}")

def show_model_comparison(results_df, problem_type):
    """Display model comparison results."""
    results_window = tk.Toplevel(root)
    results_window.title("Model Comparison Results")
    results_window.geometry("1200x800")
    results_window.configure(bg=BACKGROUND_COLOR)
    
    # Position the window
    results_window.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

    # Main container with stylish border
    main_frame = tk.Frame(results_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    title_label = tk.Label(main_frame, text="Model Comparison Results", 
                         font=("Helvetica", 18, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    title_label.pack(pady=(10, 20))

    # Results frame
    results_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # Table frame
    table_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, highlightbackground=BORDER_COLOR, highlightthickness=1)
    table_frame.pack(fill=tk.BOTH, expand=True)

    # Use pandastable to display DataFrame
    pt = Table(table_frame, dataframe=results_df, showtoolbar=False, showstatusbar=False)
    pt.show()

    # Plot frame
    plot_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if problem_type == "Classification":
        # Sort by accuracy
        results_df = results_df.sort_values("Accuracy", ascending=False)
        bars = ax.bar(results_df["Model"], results_df["Accuracy"], color=ACCENT_COLOR, alpha=0.7)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Model Comparison - Classification", fontsize=16)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        # Highlight best model
        best_idx = results_df["Accuracy"].idxmax()
        bars[best_idx].set_color(SUCCESS_COLOR)
        
    elif problem_type == "Regression":
        # Sort by R² score
        results_df = results_df.sort_values("R²", ascending=False)
        x = np.arange(len(results_df))
        width = 0.35
        
        # Plot MSE and R²
        bars1 = ax.bar(x - width/2, results_df["MSE"], width, label='MSE', color=DANGER_COLOR, alpha=0.7)
        bars2 = ax.bar(x + width/2, results_df["R²"], width, label='R²', color=SUCCESS_COLOR, alpha=0.7)
        
        ax.set_ylabel("Scores", fontsize=12)
        ax.set_title("Model Comparison - Regression", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(results_df["Model"], rotation=45)
        ax.legend()
        
        # Highlight best model (highest R²)
        best_idx = results_df["R²"].idxmax()
        bars2[best_idx].set_color(INFO_COLOR)
        
    else:  # Clustering
        # Sort by silhouette score
        results_df = results_df.sort_values("Silhouette Score", ascending=False)
        bars = ax.bar(results_df["Model"], results_df["Silhouette Score"], color=ACCENT_COLOR, alpha=0.7)
        ax.set_ylabel("Silhouette Score", fontsize=12)
        ax.set_title("Model Comparison - Clustering", fontsize=16)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        # Highlight best model
        best_idx = results_df["Silhouette Score"].idxmax()
        bars[best_idx].set_color(SUCCESS_COLOR)
    
    # Style the plot
    ax.tick_params(axis='x', rotation=45)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(2)
        
    ax.tick_params(colors=TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Embed plot
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Close button
    close_btn = tk.Button(main_frame, text="Close", command=results_window.destroy,
                         bg=DANGER_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground="#B71C1C")
    close_btn.pack(pady=(10, 20))
    add_hover_effect(close_btn, hover_color="#B71C1C", original_color=DANGER_COLOR)

# =============================================================================
# DATA PREPARATION FUNCTIONS
# =============================================================================

def show_prepare_data_interface():
    """Show interface for data preparation with improved styling."""
    prep_window = tk.Toplevel(root)
    prep_window.title("Prepare Data")
    prep_window.geometry("1100x800")
    prep_window.configure(bg=BACKGROUND_COLOR)
    
    # Position the window
    prep_window.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

    # Main container with stylish border
    main_frame = tk.Frame(prep_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Title
    title_label = tk.Label(main_frame, text="Data Preparation", 
                         font=("Helvetica", 18, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    title_label.pack(pady=(10, 20))

    # Tab control with modern style
    style = ttk.Style()
    style.configure('TNotebook', background=PRIMARY_COLOR)
    style.configure('TNotebook.Tab', background=SECONDARY_COLOR, foreground=TEXT_COLOR,
                   font=('Helvetica', 10, 'bold'), padding=[10, 5])
    style.map('TNotebook.Tab', background=[('selected', ACCENT_COLOR)], foreground=[('selected', 'white')])
    
    tab_control = ttk.Notebook(main_frame)
    tab_control.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Tab 1: Handle Missing Values
    tab1 = ttk.Frame(tab_control)
    tab_control.add(tab1, text="Handle Missing Values")
    setup_missing_values_tab(tab1)

    # Tab 2: Feature Selection
    tab2 = ttk.Frame(tab_control)
    tab_control.add(tab2, text="Feature Selection")
    setup_feature_selection_tab(tab2)

    # Tab 3: Data Type Conversion
    tab3 = ttk.Frame(tab_control)
    tab_control.add(tab3, text="Data Type Conversion")
    setup_data_type_conversion_tab(tab3)

    # Tab 4: Outlier Detection
    tab4 = ttk.Frame(tab_control)
    tab_control.add(tab4, text="Outlier Detection")
    setup_outlier_detection_tab(tab4)

    # Tab 5: PCA
    tab5 = ttk.Frame(tab_control)
    tab_control.add(tab5, text="PCA")
    setup_pca_tab(tab5)

    # Tab 6: Statistical Analysis
    tab6 = ttk.Frame(tab_control)
    tab_control.add(tab6, text="Statistical Analysis")
    setup_statistical_analysis_tab(tab6)

    # Tab 7: Dataset Info
    tab7 = ttk.Frame(tab_control)
    tab_control.add(tab7, text="Dataset Info")
    setup_dataset_info_tab(tab7)

    # Close button with modern style
    close_btn = tk.Button(main_frame, text="Close", command=prep_window.destroy,
                         bg=DANGER_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=30, pady=8, bd=0, highlightthickness=0,
                         activebackground="#B71C1C")
    close_btn.pack(pady=(10, 20))
    add_hover_effect(close_btn, hover_color="#B71C1C", original_color=DANGER_COLOR)

def setup_missing_values_tab(tab):
    """Setup the missing values tab."""
    tab.configure(style='TFrame')
    
    # Info label
    info_label = tk.Label(tab, text="Select how to handle missing values for each column:", 
                         font=("Helvetica", 12), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    info_label.pack(pady=(10, 20))

    # Get columns with missing values
    null_counts = df.isnull().sum()
    cols_with_missing = null_counts[null_counts > 0].index.tolist()
    
    if not cols_with_missing:
        no_missing_label = tk.Label(tab, text="No missing values found in the dataset!", 
                                  font=("Helvetica", 12), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
        no_missing_label.pack(pady=20)
        return

    # Scrollable frame for columns
    scroll_frame = tk.Frame(tab, bg=PRIMARY_COLOR)
    scroll_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(scroll_frame, bg=PRIMARY_COLOR, highlightthickness=0)
    scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg=PRIMARY_COLOR)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Create a frame for each column with missing values
    column_frames = []
    for col in cols_with_missing:
        col_frame = tk.Frame(scrollable_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.GROOVE, 
                           highlightbackground=BORDER_COLOR, highlightthickness=1, padx=10, pady=10)
        column_frames.append(col_frame)
        col_frame.pack(fill=tk.X, padx=10, pady=5)

        # Column name and missing count
        col_label = tk.Label(col_frame, text=f"{col} ({null_counts[col]} missing values)", 
                            font=("Helvetica", 11, "bold"), fg=TEXT_COLOR, bg=SECONDARY_COLOR)
        col_label.pack(anchor=tk.W)

        # Radio buttons for handling options
        option_var = tk.StringVar(value="drop")
        
        drop_radio = tk.Radiobutton(col_frame, text="Drop rows with missing values", 
                                   variable=option_var, value="drop", 
                                   bg=SECONDARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                                   activebackground=SECONDARY_COLOR, activeforeground=TEXT_COLOR,
                                   font=("Helvetica", 10))
        drop_radio.pack(anchor=tk.W)

        mean_radio = tk.Radiobutton(col_frame, text="Fill with mean value", 
                                   variable=option_var, value="mean", 
                                   bg=SECONDARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                                   activebackground=SECONDARY_COLOR, activeforeground=TEXT_COLOR,
                                   font=("Helvetica", 10))
        mean_radio.pack(anchor=tk.W)

        median_radio = tk.Radiobutton(col_frame, text="Fill with median value", 
                                     variable=option_var, value="median", 
                                     bg=SECONDARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                                     activebackground=SECONDARY_COLOR, activeforeground=TEXT_COLOR,
                                     font=("Helvetica", 10))
        median_radio.pack(anchor=tk.W)

        mode_radio = tk.Radiobutton(col_frame, text="Fill with mode value", 
                                   variable=option_var, value="mode", 
                                   bg=SECONDARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                                   activebackground=SECONDARY_COLOR, activeforeground=TEXT_COLOR,
                                   font=("Helvetica", 10))
        mode_radio.pack(anchor=tk.W)

        zero_radio = tk.Radiobutton(col_frame, text="Fill with 0", 
                                   variable=option_var, value="zero", 
                                   bg=SECONDARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                                   activebackground=SECONDARY_COLOR, activeforeground=TEXT_COLOR,
                                   font=("Helvetica", 10))
        zero_radio.pack(anchor=tk.W)

        # Store the column and its option variable
        col_frame.column_name = col
        col_frame.option_var = option_var

    # Apply button
    apply_btn = tk.Button(tab, text="Apply Changes", command=lambda: apply_missing_value_changes(column_frames),
                         bg=SUCCESS_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground=HOVER_COLOR)
    apply_btn.pack(pady=(20, 10))
    add_hover_effect(apply_btn, hover_color=HOVER_COLOR, original_color=SUCCESS_COLOR)

def apply_missing_value_changes(column_frames):
    """Apply the selected missing value handling methods."""
    global df
    
    try:
        for col_frame in column_frames:
            col = col_frame.column_name
            option = col_frame.option_var.get()
            
            if option == "drop":
                df = df.dropna(subset=[col])
            elif option == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif option == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif option == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif option == "zero":
                df[col].fillna(0, inplace=True)
        
        messagebox.showinfo("Success", "Missing value handling applied successfully!")
        update_data_preview()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to handle missing values: {e}")

def setup_feature_selection_tab(tab):
    """Setup the feature selection tab."""
    tab.configure(style='TFrame')
    
    # Info label
    info_label = tk.Label(tab, text="Select features and target variable(s):", 
                         font=("Helvetica", 12), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    info_label.pack(pady=(10, 20))

    # Main frame for feature selection
    selection_frame = tk.Frame(tab, bg=PRIMARY_COLOR)
    selection_frame.pack(fill=tk.BOTH, expand=True)

    # Available features frame
    available_frame = tk.Frame(selection_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, 
                             highlightbackground=BORDER_COLOR, highlightthickness=1)
    available_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

    available_label = tk.Label(available_frame, text="Available Features", 
                             font=("Helvetica", 11, "bold"), fg=TEXT_COLOR, bg=SECONDARY_COLOR)
    available_label.pack(pady=5)

    # Listbox for available features
    available_list = tk.Listbox(available_frame, selectmode=tk.MULTIPLE, 
                              bg="white", fg=TEXT_COLOR, font=("Helvetica", 10),
                              selectbackground=ACCENT_COLOR, selectforeground="white",
                              highlightthickness=0, bd=0)
    scrollbar = ttk.Scrollbar(available_frame, orient="vertical", command=available_list.yview)
    available_list.configure(yscrollcommand=scrollbar.set)
    
    # Populate list with columns
    for col in df.columns:
        available_list.insert(tk.END, col)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    available_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Buttons frame
    button_frame = tk.Frame(selection_frame, bg=PRIMARY_COLOR)
    button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

    # Add to features button
    add_feature_btn = tk.Button(button_frame, text="→ Features →", 
                              command=lambda: move_items(available_list, features_list),
                              bg=ACCENT_COLOR, fg="white", font=("Helvetica", 10, "bold"),
                              relief=tk.FLAT, padx=10, pady=5, bd=0, highlightthickness=0,
                              activebackground=HOVER_COLOR)
    add_feature_btn.pack(pady=10)
    add_hover_effect(add_feature_btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

    # Add to target button
    add_target_btn = tk.Button(button_frame, text="→ Target →", 
                             command=lambda: move_items(available_list, target_list, allow_multiple=False),
                             bg=ACCENT_COLOR, fg="white", font=("Helvetica", 10, "bold"),
                             relief=tk.FLAT, padx=10, pady=5, bd=0, highlightthickness=0,
                             activebackground=HOVER_COLOR)
    add_target_btn.pack(pady=10)
    add_hover_effect(add_target_btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

    # Remove button
    remove_btn = tk.Button(button_frame, text="← Remove ←", 
                          command=lambda: remove_items(features_list, target_list),
                          bg=DANGER_COLOR, fg="white", font=("Helvetica", 10, "bold"),
                          relief=tk.FLAT, padx=10, pady=5, bd=0, highlightthickness=0,
                          activebackground="#B71C1C")
    remove_btn.pack(pady=10)
    add_hover_effect(remove_btn, hover_color="#B71C1C", original_color=DANGER_COLOR)

    # Selected features frame
    selected_frame = tk.Frame(selection_frame, bg=PRIMARY_COLOR)
    selected_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Features list frame
    features_frame = tk.Frame(selected_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, 
                            highlightbackground=BORDER_COLOR, highlightthickness=1)
    features_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    features_label = tk.Label(features_frame, text="Input Features", 
                            font=("Helvetica", 11, "bold"), fg=TEXT_COLOR, bg=SECONDARY_COLOR)
    features_label.pack(pady=5)

    features_list = tk.Listbox(features_frame, selectmode=tk.MULTIPLE, 
                             bg="white", fg=TEXT_COLOR, font=("Helvetica", 10),
                             selectbackground=ACCENT_COLOR, selectforeground="white",
                             highlightthickness=0, bd=0)
    features_scroll = ttk.Scrollbar(features_frame, orient="vertical", command=features_list.yview)
    features_list.configure(yscrollcommand=features_scroll.set)
    
    features_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    features_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Target frame
    target_frame = tk.Frame(selected_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, 
                          highlightbackground=BORDER_COLOR, highlightthickness=1)
    target_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    target_label = tk.Label(target_frame, text="Target Variable", 
                          font=("Helvetica", 11, "bold"), fg=TEXT_COLOR, bg=SECONDARY_COLOR)
    target_label.pack(pady=5)

    target_list = tk.Listbox(target_frame, selectmode=tk.SINGLE, 
                           bg="white", fg=TEXT_COLOR, font=("Helvetica", 10),
                           selectbackground=ACCENT_COLOR, selectforeground="white",
                           highlightthickness=0, bd=0)
    target_scroll = ttk.Scrollbar(target_frame, orient="vertical", command=target_list.yview)
    target_list.configure(yscrollcommand=target_scroll.set)
    
    target_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    target_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Apply button
    apply_btn = tk.Button(tab, text="Apply Feature Selection", 
                        command=lambda: apply_feature_selection(features_list, target_list),
                        bg=SUCCESS_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                        relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                        activebackground=HOVER_COLOR)
    apply_btn.pack(pady=(20, 10))
    add_hover_effect(apply_btn, hover_color=HOVER_COLOR, original_color=SUCCESS_COLOR)

def move_items(source_list, dest_list, allow_multiple=True):
    """Move selected items from source list to destination list."""
    selected = source_list.curselection()
    if not selected:
        return
    
    if not allow_multiple and dest_list.size() > 0:
        messagebox.showwarning("Warning", "Only one target variable is allowed")
        return
        
    for idx in selected[::-1]:
        item = source_list.get(idx)
        dest_list.insert(tk.END, item)
        source_list.delete(idx)

def remove_items(*lists):
    """Remove selected items from lists and return them to available features."""
    for lst in lists:
        selected = lst.curselection()
        for idx in selected[::-1]:
            item = lst.get(idx)
            available_list = root.nametowidget(lst.master.master.master.children['!frame'].children['!listbox'])
            available_list.insert(tk.END, item)
            lst.delete(idx)

def apply_feature_selection(features_list, target_list):
    """Apply the selected feature and target variables."""
    global df, X, y
    
    try:
        # Get selected features and target
        features = [features_list.get(i) for i in range(features_list.size())]
        targets = [target_list.get(i) for i in range(target_list.size())]
        
        if not features:
            messagebox.showerror("Error", "Please select at least one feature")
            return
            
        if not targets:
            messagebox.showerror("Error", "Please select at least one target variable")
            return
            
        # Update the dataframe to only include selected columns
        df = df[features + targets]
        
        # Set X and y
        X = df[features]
        if len(targets) == 1:
            y = df[targets[0]]
        else:
            y = df[targets]
            
        messagebox.showinfo("Success", "Feature selection applied successfully!")
        update_data_preview()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to apply feature selection: {e}")

def setup_data_type_conversion_tab(tab):
    """Setup the data type conversion tab with comprehensive options."""
    tab.configure(style='TFrame')
    
    # Info label
    info_label = tk.Label(tab, text="Convert data types for columns:", 
                         font=("Helvetica", 12), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    info_label.pack(pady=(10, 20))

    # Main frame
    main_frame = tk.Frame(tab, bg=PRIMARY_COLOR)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Available columns frame
    columns_frame = tk.Frame(main_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, 
                           highlightbackground=BORDER_COLOR, highlightthickness=1)
    columns_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    columns_label = tk.Label(columns_frame, text="Select Columns", 
                           font=("Helvetica", 11, "bold"), fg=TEXT_COLOR, bg=SECONDARY_COLOR)
    columns_label.pack(pady=5)

    # Listbox for columns
    columns_list = tk.Listbox(columns_frame, selectmode=tk.MULTIPLE, 
                            bg="white", fg=TEXT_COLOR, font=("Helvetica", 10),
                            selectbackground=ACCENT_COLOR, selectforeground="white",
                            highlightthickness=0, bd=0)
    scrollbar = ttk.Scrollbar(columns_frame, orient="vertical", command=columns_list.yview)
    columns_list.configure(yscrollcommand=scrollbar.set)
    
    # Populate list with columns and their current data types
    for col in df.columns:
        columns_list.insert(tk.END, f"{col} ({df[col].dtype})")
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    columns_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Conversion options frame
    conversion_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    conversion_frame.pack(fill=tk.X, padx=10, pady=10)

    # Conversion type
    conversion_label = tk.Label(conversion_frame, text="Convert to:", 
                              font=("Helvetica", 11, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    conversion_label.pack(anchor=tk.W)

    conversion_var = tk.StringVar(value="numeric_float")
    
    # Comprehensive conversion options
    conversion_options = [
        ("Float (numeric)", "numeric_float"),
        ("Integer (numeric)", "numeric_int"),
        ("Categorical", "categorical"),
        ("String (object)", "string"),
        ("Boolean", "boolean"),
        ("Datetime", "datetime")
    ]
    
    for text, value in conversion_options:
        radio = tk.Radiobutton(conversion_frame, text=text, 
                              variable=conversion_var, value=value, 
                              bg=PRIMARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                              activebackground=PRIMARY_COLOR, activeforeground=TEXT_COLOR,
                              font=("Helvetica", 10))
        radio.pack(anchor=tk.W)

    # Additional options for numeric conversion
    options_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    options_frame.pack(fill=tk.X, padx=10, pady=5)

    # Handle errors option for numeric conversion
    error_handling_var = tk.StringVar(value="coerce")
    error_label = tk.Label(options_frame, text="Error handling for numeric conversion:", 
                         font=("Helvetica", 10), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    error_label.pack(anchor=tk.W)
    
    coerce_radio = tk.Radiobutton(options_frame, text="Coerce errors to NaN", 
                                 variable=error_handling_var, value="coerce", 
                                 bg=PRIMARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                                 activebackground=PRIMARY_COLOR, activeforeground=TEXT_COLOR,
                                 font=("Helvetica", 9))
    coerce_radio.pack(anchor=tk.W)
    
    ignore_radio = tk.Radiobutton(options_frame, text="Ignore errors (keep original)", 
                                 variable=error_handling_var, value="ignore", 
                                 bg=PRIMARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                                 activebackground=PRIMARY_COLOR, activeforeground=TEXT_COLOR,
                                 font=("Helvetica", 9))
    ignore_radio.pack(anchor=tk.W)

    # Apply button
    apply_btn = tk.Button(main_frame, text="Apply Conversion", 
                        command=lambda: apply_data_type_conversion(columns_list, conversion_var.get(), error_handling_var.get()),
                        bg=SUCCESS_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                        relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                        activebackground=HOVER_COLOR)
    apply_btn.pack(pady=(10, 20))
    add_hover_effect(apply_btn, hover_color=HOVER_COLOR, original_color=SUCCESS_COLOR)

def apply_data_type_conversion(columns_list, conversion_type, error_handling="coerce"):
    """Apply data type conversion to selected columns with comprehensive options."""
    global df
    
    try:
        selected = columns_list.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select at least one column")
            return
        
        conversion_results = []
        for idx in selected:
            col_name = columns_list.get(idx).split(" (")[0]
            original_dtype = str(df[col_name].dtype)
            
            try:
                if conversion_type == "numeric_float":
                    df[col_name] = pd.to_numeric(df[col_name], errors=error_handling)
                    new_dtype = str(df[col_name].dtype)
                    
                elif conversion_type == "numeric_int":
                    # First convert to float, then to int (handles NaN better)
                    temp_series = pd.to_numeric(df[col_name], errors=error_handling)
                    # Only convert non-NaN values to int
                    df[col_name] = temp_series.apply(lambda x: int(x) if pd.notna(x) else x)
                    new_dtype = str(df[col_name].dtype)
                    
                elif conversion_type == "categorical":
                    df[col_name] = df[col_name].astype('category')
                    new_dtype = "category"
                    
                elif conversion_type == "string":
                    df[col_name] = df[col_name].astype(str)
                    new_dtype = "string"
                    
                elif conversion_type == "boolean":
                    # Convert common representations to boolean
                    if df[col_name].dtype == 'object':
                        # Handle string representations
                        true_values = ['true', 'True', 'TRUE', '1', 'yes', 'Yes', 'YES']
                        false_values = ['false', 'False', 'FALSE', '0', 'no', 'No', 'NO']
                        
                        def to_boolean(val):
                            if str(val).strip() in true_values:
                                return True
                            elif str(val).strip() in false_values:
                                return False
                            else:
                                return pd.NA
                        
                        df[col_name] = df[col_name].apply(to_boolean)
                    else:
                        # For numeric, convert to boolean
                        df[col_name] = df[col_name].astype(bool)
                    new_dtype = "boolean"
                    
                elif conversion_type == "datetime":
                    df[col_name] = pd.to_datetime(df[col_name], errors=error_handling)
                    new_dtype = str(df[col_name].dtype)
                
                conversion_results.append(f"{col_name}: {original_dtype} → {new_dtype}")
                
            except Exception as col_error:
                conversion_results.append(f"{col_name}: Error - {str(col_error)}")
        
        # Show conversion results
        result_message = "Conversion Results:\n\n" + "\n".join(conversion_results)
        messagebox.showinfo("Conversion Complete", result_message)
        update_data_preview()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to convert data types: {e}")


def setup_outlier_detection_tab(tab):
    """Setup the outlier detection tab."""
    tab.configure(style='TFrame')
    
    # Info label
    info_label = tk.Label(tab, text="Detect and handle outliers in numeric columns:", 
                         font=("Helvetica", 12), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    info_label.pack(pady=(10, 20))

    # Main frame
    main_frame = tk.Frame(tab, bg=PRIMARY_COLOR)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Numeric columns frame
    numeric_frame = tk.Frame(main_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, 
                           highlightbackground=BORDER_COLOR, highlightthickness=1)
    numeric_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    numeric_label = tk.Label(numeric_frame, text="Numeric Columns", 
                           font=("Helvetica", 11, "bold"), fg=TEXT_COLOR, bg=SECONDARY_COLOR)
    numeric_label.pack(pady=5)

    # Listbox for numeric columns
    numeric_list = tk.Listbox(numeric_frame, selectmode=tk.MULTIPLE, 
                            bg="white", fg=TEXT_COLOR, font=("Helvetica", 10),
                            selectbackground=ACCENT_COLOR, selectforeground="white",
                            highlightthickness=0, bd=0)
    scrollbar = ttk.Scrollbar(numeric_frame, orient="vertical", command=numeric_list.yview)
    numeric_list.configure(yscrollcommand=scrollbar.set)
    
    # Populate list with numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        numeric_list.insert(tk.END, col)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    numeric_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Detection method frame
    method_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    method_frame.pack(fill=tk.X, padx=10, pady=10)

    method_label = tk.Label(method_frame, text="Detection Method:", 
                          font=("Helvetica", 11, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    method_label.pack(anchor=tk.W)

    method_var = tk.StringVar(value="iqr")
    
    iqr_radio = tk.Radiobutton(method_frame, text="IQR (Interquartile Range)", 
                              variable=method_var, value="iqr", 
                              bg=PRIMARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                              activebackground=PRIMARY_COLOR, activeforeground=TEXT_COLOR,
                              font=("Helvetica", 10))
    iqr_radio.pack(anchor=tk.W)

    zscore_radio = tk.Radiobutton(method_frame, text="Z-Score", 
                                variable=method_var, value="zscore", 
                                bg=PRIMARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                                activebackground=PRIMARY_COLOR, activeforeground=TEXT_COLOR,
                                font=("Helvetica", 10))
    zscore_radio.pack(anchor=tk.W)

    isolation_radio = tk.Radiobutton(method_frame, text="Isolation Forest", 
                                   variable=method_var, value="isolation", 
                                   bg=PRIMARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                                   activebackground=PRIMARY_COLOR, activeforeground=TEXT_COLOR,
                                   font=("Helvetica", 10))
    isolation_radio.pack(anchor=tk.W)

    # Action frame
    action_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    action_frame.pack(fill=tk.X, padx=10, pady=5)

    action_label = tk.Label(action_frame, text="Action:", 
                          font=("Helvetica", 11, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    action_label.pack(anchor=tk.W)

    action_var = tk.StringVar(value="remove")
    
    remove_radio = tk.Radiobutton(action_frame, text="Remove outliers", 
                                variable=action_var, value="remove", 
                                bg=PRIMARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                                activebackground=PRIMARY_COLOR, activeforeground=TEXT_COLOR,
                                font=("Helvetica", 10))
    remove_radio.pack(anchor=tk.W)

    cap_radio = tk.Radiobutton(action_frame, text="Cap outliers", 
                             variable=action_var, value="cap", 
                             bg=PRIMARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                             activebackground=PRIMARY_COLOR, activeforeground=TEXT_COLOR,
                             font=("Helvetica", 10))
    cap_radio.pack(anchor=tk.W)

    # Buttons frame
    buttons_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    buttons_frame.pack(fill=tk.X, padx=10, pady=10)

    # Detect button
    detect_btn = tk.Button(buttons_frame, text="Detect Outliers", 
                         command=lambda: detect_outliers(numeric_list, method_var.get()),
                         bg=INFO_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground=HOVER_COLOR)
    detect_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(detect_btn, hover_color=HOVER_COLOR, original_color=INFO_COLOR)

    # Handle button
    handle_btn = tk.Button(buttons_frame, text="Handle Outliers", 
                         command=lambda: handle_outliers(numeric_list, method_var.get(), action_var.get()),
                         bg=SUCCESS_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground=HOVER_COLOR)
    handle_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(handle_btn, hover_color=HOVER_COLOR, original_color=SUCCESS_COLOR)

def detect_outliers(numeric_list, method):
    """Detect outliers in selected columns."""
    try:
        selected = numeric_list.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select at least one numeric column")
            return
        
        results = []
        for idx in selected:
            col_name = numeric_list.get(idx)
            data = df[col_name].dropna()
            
            if method == "iqr":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(data))
                outliers = data[z_scores > 3]
                
            elif method == "isolation":
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                preds = iso_forest.fit_predict(data.values.reshape(-1, 1))
                outliers = data[preds == -1]
            
            results.append(f"{col_name}: {len(outliers)} outliers detected")
        
        messagebox.showinfo("Outlier Detection Results", "\n".join(results))
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to detect outliers: {e}")

def handle_outliers(numeric_list, method, action):
    """Handle outliers in selected columns."""
    global df
    
    try:
        selected = numeric_list.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select at least one numeric column")
            return
        
        for idx in selected:
            col_name = numeric_list.get(idx)
            data = df[col_name].copy()
            
            if method == "iqr":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(data))
                lower_bound = data.mean() - 3 * data.std()
                upper_bound = data.mean() + 3 * data.std()
                
            elif method == "isolation":
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                preds = iso_forest.fit_predict(data.values.reshape(-1, 1))
                lower_bound = data.min()
                upper_bound = data.max()
            
            if action == "remove":
                df = df[(data >= lower_bound) & (data <= upper_bound)]
            elif action == "cap":
                df[col_name] = np.where(data < lower_bound, lower_bound, data)
                df[col_name] = np.where(data > upper_bound, upper_bound, data)
        
        messagebox.showinfo("Success", "Outlier handling applied successfully!")
        update_data_preview()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to handle outliers: {e}")

def setup_pca_tab(tab):
    """Setup the PCA tab."""
    tab.configure(style='TFrame')
    
    # Info label
    info_label = tk.Label(tab, text="Principal Component Analysis (PCA):", 
                         font=("Helvetica", 12), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    info_label.pack(pady=(10, 20))

    # Main frame
    main_frame = tk.Frame(tab, bg=PRIMARY_COLOR)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Components frame
    components_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    components_frame.pack(fill=tk.X, padx=10, pady=10)

    components_label = tk.Label(components_frame, text="Number of Components:", 
                              font=("Helvetica", 11, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    components_label.pack(anchor=tk.W)

    components_var = tk.StringVar(value="2")
    components_entry = tk.Entry(components_frame, textvariable=components_var, 
                              font=("Helvetica", 10), bg="white", fg=TEXT_COLOR,
                              relief=tk.SOLID, bd=1)
    components_entry.pack(anchor=tk.W, pady=5)

    # Features frame
    features_frame = tk.Frame(main_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN, 
                            highlightbackground=BORDER_COLOR, highlightthickness=1)
    features_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    features_label = tk.Label(features_frame, text="Select Features for PCA", 
                            font=("Helvetica", 11, "bold"), fg=TEXT_COLOR, bg=SECONDARY_COLOR)
    features_label.pack(pady=5)

    # Listbox for features
    features_list = tk.Listbox(features_frame, selectmode=tk.MULTIPLE, 
                             bg="white", fg=TEXT_COLOR, font=("Helvetica", 10),
                             selectbackground=ACCENT_COLOR, selectforeground="white",
                             highlightthickness=0, bd=0)
    scrollbar = ttk.Scrollbar(features_frame, orient="vertical", command=features_list.yview)
    features_list.configure(yscrollcommand=scrollbar.set)
    
    # Populate list with numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        features_list.insert(tk.END, col)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    features_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Buttons frame
    buttons_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    buttons_frame.pack(fill=tk.X, padx=10, pady=10)

    # Apply PCA button
    apply_btn = tk.Button(buttons_frame, text="Apply PCA", 
                        command=lambda: apply_pca(features_list, components_var.get()),
                        bg=SUCCESS_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                        relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                        activebackground=HOVER_COLOR)
    apply_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(apply_btn, hover_color=HOVER_COLOR, original_color=SUCCESS_COLOR)

    # Visualize button
    visualize_btn = tk.Button(buttons_frame, text="Visualize PCA", 
                            command=lambda: visualize_pca(features_list, components_var.get()),
                            bg=INFO_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                            relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                            activebackground=HOVER_COLOR)
    visualize_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(visualize_btn, hover_color=HOVER_COLOR, original_color=INFO_COLOR)

def apply_pca(features_list, n_components):
    """Apply PCA to selected features."""
    global df
    
    try:
        selected = features_list.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select at least one feature")
            return
        
        n_components = int(n_components)
        if n_components <= 0:
            messagebox.showerror("Error", "Number of components must be positive")
            return
        
        # Get selected features
        features = [features_list.get(i) for i in selected]
        X = df[features].dropna()
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create new column names
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        
        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        # Add PCA components to original DataFrame
        for col in pca_columns:
            df[col] = pca_df[col]
        
        # Show explained variance
        explained_variance = pca.explained_variance_ratio_
        variance_info = "\n".join([f"PC{i+1}: {var:.2%}" for i, var in enumerate(explained_variance)])
        
        messagebox.showinfo("PCA Results", 
                          f"PCA applied successfully!\n\nExplained Variance:\n{variance_info}\n\nTotal: {explained_variance.sum():.2%}")
        update_data_preview()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to apply PCA: {e}")

def visualize_pca(features_list, n_components):
    """Visualize PCA results."""
    try:
        selected = features_list.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select at least one feature")
            return
        
        n_components = int(n_components)
        if n_components < 2:
            messagebox.showerror("Error", "Need at least 2 components for visualization")
            return
        
        # Get selected features
        features = [features_list.get(i) for i in selected]
        X = df[features].dropna()
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create visualization window
        viz_window = tk.Toplevel(root)
        viz_window.title("PCA Visualization")
        viz_window.geometry("800x600")
        viz_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot of first two components
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, color=ACCENT_COLOR)
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.set_title('PCA - First Two Components')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Explained variance plot
        explained_variance = pca.explained_variance_ratio_
        components = range(1, len(explained_variance) + 1)
        ax2.bar(components, explained_variance, color=ACCENT_COLOR, alpha=0.7)
        ax2.set_xlabel('Principal Components')
        ax2.set_ylabel('Explained Variance Ratio')
        ax2.set_title('Explained Variance by Component')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=viz_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to visualize PCA: {e}")

def setup_statistical_analysis_tab(tab):
    """Setup the statistical analysis tab."""
    tab.configure(style='TFrame')
    
    # Info label
    info_label = tk.Label(tab, text="Statistical Analysis of Dataset", 
                         font=("Helvetica", 12), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    info_label.pack(pady=(10, 20))

    # Analysis frame
    analysis_frame = tk.Frame(tab, bg=PRIMARY_COLOR)
    analysis_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # Text widget for statistical summary
    stats_text = scrolledtext.ScrolledText(analysis_frame, wrap=tk.WORD, 
                                         bg=SECONDARY_COLOR, fg=TEXT_COLOR, 
                                         insertbackground=TEXT_COLOR, 
                                         font=("Courier", 10),
                                         padx=10, pady=10)
    stats_text.pack(fill=tk.BOTH, expand=True)

    # Generate statistical summary
    try:
        stats_text.insert(tk.END, "DATASET STATISTICAL SUMMARY\n\n", "header")
        
        # Basic info
        stats_text.insert(tk.END, f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number])
        if not numeric_cols.empty:
            stats_text.insert(tk.END, "NUMERIC COLUMNS SUMMARY:\n\n")
            stats_text.insert(tk.END, numeric_cols.describe().to_string())
            stats_text.insert(tk.END, "\n\n")
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object'])
        if not categorical_cols.empty:
            stats_text.insert(tk.END, "CATEGORICAL COLUMNS SUMMARY:\n\n")
            for col in categorical_cols.columns:
                stats_text.insert(tk.END, f"{col}:\n")
                stats_text.insert(tk.END, f"  Unique values: {df[col].nunique()}\n")
                stats_text.insert(tk.END, f"  Most frequent: {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}\n")
                stats_text.insert(tk.END, f"  Frequency: {df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0}\n\n")
        
        # Missing values
        stats_text.insert(tk.END, "MISSING VALUES:\n\n")
        missing_counts = df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                stats_text.insert(tk.END, f"  {col}: {count} ({count/len(df)*100:.1f}%)\n")
        
        if missing_counts.sum() == 0:
            stats_text.insert(tk.END, "  No missing values found\n")
            
    except Exception as e:
        stats_text.insert(tk.END, f"Error generating statistical summary: {e}")
    
    stats_text.config(state=tk.DISABLED)

def setup_dataset_info_tab(tab):
    """Setup the dataset information tab."""
    tab.configure(style='TFrame')
    
    # Info label
    info_label = tk.Label(tab, text="Dataset Information", 
                         font=("Helvetica", 12), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    info_label.pack(pady=(10, 20))

    # Info frame
    info_frame = tk.Frame(tab, bg=PRIMARY_COLOR)
    info_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # Text widget for dataset info
    info_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, 
                                        bg=SECONDARY_COLOR, fg=TEXT_COLOR, 
                                        insertbackground=TEXT_COLOR, 
                                        font=("Courier", 10),
                                        padx=10, pady=10)
    info_text.pack(fill=tk.BOTH, expand=True)

    # Generate dataset info
    try:
        info_text.insert(tk.END, "DATASET INFORMATION\n\n", "header")
        
        # Basic info
        info_text.insert(tk.END, f"Number of rows: {df.shape[0]}\n")
        info_text.insert(tk.END, f"Number of columns: {df.shape[1]}\n")
        info_text.insert(tk.END, f"Total cells: {df.size}\n\n")
        
        # Memory usage
        info_text.insert(tk.END, f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        # Data types
        info_text.insert(tk.END, "DATA TYPES:\n\n")
        dtypes_summary = df.dtypes.value_counts()
        for dtype, count in dtypes_summary.items():
            info_text.insert(tk.END, f"  {dtype}: {count} columns\n")
        info_text.insert(tk.END, "\n")
        
        # Column information
        info_text.insert(tk.END, "COLUMN DETAILS:\n\n")
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].count()
            null_count = df[col].isnull().sum()
            info_text.insert(tk.END, f"  {col} ({dtype}): {non_null} non-null, {null_count} null\n")
            
    except Exception as e:
        info_text.insert(tk.END, f"Error generating dataset info: {e}")
    
    info_text.config(state=tk.DISABLED)

# =============================================================================
# STANDARDIZATION FUNCTION
# =============================================================================

def standardize_data():
    """Standardize the dataset using StandardScaler."""
    global df
    
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number])
        if numeric_cols.empty:
            messagebox.showerror("Error", "No numeric columns found to standardize!")
            return
        
        # Apply StandardScaler
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(numeric_cols)
        
        # Create new column names
        scaled_cols = [f"{col}_scaled" for col in numeric_cols.columns]
        
        # Create DataFrame with scaled values
        scaled_df = pd.DataFrame(df_scaled, columns=scaled_cols, index=df.index)
        
        # Add scaled columns to original DataFrame
        for col in scaled_cols:
            df[col] = scaled_df[col]
        
        messagebox.showinfo("Success", "Data standardized successfully! Scaled columns added with '_scaled' suffix.")
        update_data_preview()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to standardize data: {e}")

# =============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# =============================================================================

def show_advanced_analysis():
    """Show advanced analysis options (SHAP and Ensemble methods)."""
    popup = tk.Toplevel(root)
    popup.title("Advanced Analysis")
    popup.geometry("600x500")
    popup.configure(bg=BACKGROUND_COLOR)
    
    # Position the popup
    popup.geometry(f"+{root.winfo_x()+root.winfo_width()//2-300}+{root.winfo_y()+root.winfo_height()//2-250}")

    # Main container with stylish border
    main_frame = tk.Frame(popup, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    title_label = tk.Label(main_frame, text="Advanced Analysis", 
                         font=("Helvetica", 18, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    title_label.pack(pady=(10, 20))

    # Advanced analysis options
    advanced_options = [
        ("SHAP Analysis - Summary Plot", run_shap_analysis),
        ("SHAP Analysis - Dependency Plot", run_shap_dependency),
        ("Ensemble Method Comparison", compare_ensemble_methods),
        ("Feature Importance Analysis", run_feature_importance)
    ]
    
    for option_name, option_func in advanced_options:
        btn = tk.Button(main_frame, text=option_name, command=option_func,
                       bg=ACCENT_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                       relief=tk.FLAT, padx=20, pady=12, bd=0, highlightthickness=0,
                       activebackground=HOVER_COLOR, width=30)
        btn.pack(pady=8)
        add_hover_effect(btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

    # Close button
    close_btn = tk.Button(main_frame, text="Close", command=popup.destroy,
                         bg=DANGER_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground="#B71C1C")
    close_btn.pack(pady=(20, 10))
    add_hover_effect(close_btn, hover_color="#B71C1C", original_color=DANGER_COLOR)

# def run_shap_analysis():
#     """Run SHAP analysis and show summary plot."""
#     try:
#         if df.empty:
#             messagebox.showerror("Error", "No data loaded!")
#             return
        
#         # Get data
#         X = df.iloc[:, :-1]
#         y = df.iloc[:, -1]
        
#         # Encode categorical target if needed
#         if y.dtype == 'object':
#             le = LabelEncoder()
#             y = le.fit_transform(y)
        
#         # Train a model
#         model = RandomForestClassifier(n_estimators=100, random_state=42)
#         model.fit(X, y)
        
#         # Calculate SHAP values
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(X)
        
#         # Create visualization window
#         shap_window = tk.Toplevel(root)
#         shap_window.title("SHAP Analysis - Summary Plot")
#         shap_window.geometry("1000x800")
#         shap_window.configure(bg=BACKGROUND_COLOR)
        
#         # Create figure
#         fig, ax = plt.subplots(figsize=(10, 8))
#         shap.summary_plot(shap_values, X, show=False)
#         plt.tight_layout()
        
#         # Display plot
#         canvas = FigureCanvasTkAgg(fig, master=shap_window)
#         canvas.draw()
#         canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
#     except Exception as e:
#         messagebox.showerror("Error", f"Failed to run SHAP analysis: {e}")

def run_shap_analysis():
    """Run SHAP analysis and show summary plot with proper error handling."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Get data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Check if we have enough data
        if len(X) < 10:
            messagebox.showerror("Error", "Not enough data for SHAP analysis (need at least 10 samples)")
            return
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y
        
        # Handle categorical features in X
        X_processed = X.copy()
        categorical_features = X_processed.select_dtypes(include=['object', 'category']).columns
        
        if not categorical_features.empty:
            # One-hot encode categorical features
            X_processed = pd.get_dummies(X_processed, columns=categorical_features, drop_first=True)
        
        # Remove any remaining non-numeric columns
        X_processed = X_processed.select_dtypes(include=[np.number])
        
        if X_processed.empty:
            messagebox.showerror("Error", "No numeric features available for SHAP analysis after preprocessing")
            return
        
        # Check for NaN values
        if X_processed.isnull().any().any():
            messagebox.showwarning("Warning", "NaN values found in features. They will be filled with mean values.")
            X_processed = X_processed.fillna(X_processed.mean())
        
        # Train a model (using RandomForest for better SHAP compatibility)
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_processed, y_encoded)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values (use a subset if dataset is large)
        if len(X_processed) > 1000:
            messagebox.showinfo("Info", "Using subset of 1000 samples for SHAP analysis")
            X_sample = X_processed.sample(n=1000, random_state=42)
            shap_values = explainer.shap_values(X_sample)
        else:
            shap_values = explainer.shap_values(X_processed)
        
        # Create visualization window
        shap_window = tk.Toplevel(root)
        shap_window.title("SHAP Analysis - Summary Plot")
        shap_window.geometry("1200x800")
        shap_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Summary plot
        if isinstance(shap_values, list):
            # For multi-class classification, use the first class
            shap.summary_plot(shap_values[0], X_processed, show=False, plot_type="dot", ax=ax1)
        else:
            # For binary classification or regression
            shap.summary_plot(shap_values, X_processed, show=False, plot_type="dot", ax=ax1)
        
        ax1.set_title("SHAP Summary Plot", fontsize=16)
        
        # Bar plot (mean absolute SHAP values)
        if isinstance(shap_values, list):
            shap_means = np.abs(shap_values[0]).mean(axis=0)
        else:
            shap_means = np.abs(shap_values).mean(axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'importance': shap_means
        }).sort_values('importance', ascending=True)
        
        ax2.barh(feature_importance['feature'], feature_importance['importance'], color=ACCENT_COLOR)
        ax2.set_xlabel('Mean |SHAP value|')
        ax2.set_title('Feature Importance (SHAP)', fontsize=16)
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=shap_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add info text
        info_text = f"SHAP Analysis completed successfully!\n\n"
        info_text += f"Dataset shape: {X_processed.shape}\n"
        info_text += f"Number of features: {len(X_processed.columns)}\n"
        info_text += f"Target type: {'Classification' if len(np.unique(y_encoded)) > 1 else 'Regression'}"
        
        info_label = tk.Label(shap_window, text=info_text, font=("Helvetica", 10), 
                            bg=SECONDARY_COLOR, fg=TEXT_COLOR, justify=tk.LEFT)
        info_label.pack(pady=10)
        
    except Exception as e:
        error_msg = f"Failed to run SHAP analysis: {str(e)}\n\n"
        error_msg += "Common issues:\n"
        error_msg += "1. Not enough samples\n"
        error_msg += "2. All features are categorical\n"
        error_msg += "3. Too many NaN values\n"
        error_msg += "4. Memory limitations with large datasets"
        messagebox.showerror("SHAP Analysis Error", error_msg)

def run_shap_dependency():
    """Run SHAP dependency plot with proper feature selection."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Get data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Check if we have enough data
        if len(X) < 10:
            messagebox.showerror("Error", "Not enough data for SHAP analysis")
            return
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y
        
        # Handle categorical features
        X_processed = X.copy()
        categorical_features = X_processed.select_dtypes(include=['object', 'category']).columns
        
        if not categorical_features.empty:
            X_processed = pd.get_dummies(X_processed, columns=categorical_features, drop_first=True)
        
        # Remove any remaining non-numeric columns
        X_processed = X_processed.select_dtypes(include=[np.number])
        
        if X_processed.empty:
            messagebox.showerror("Error", "No numeric features available for SHAP analysis")
            return
        
        # Handle NaN values
        if X_processed.isnull().any().any():
            X_processed = X_processed.fillna(X_processed.mean())
        
        # Train a model
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_processed, y_encoded)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        if len(X_processed) > 1000:
            X_sample = X_processed.sample(n=1000, random_state=42)
            shap_values = explainer.shap_values(X_sample)
        else:
            shap_values = explainer.shap_values(X_processed)
        
        # Let user select feature for dependency plot
        feature_window = tk.Toplevel(root)
        feature_window.title("Select Feature for SHAP Dependency Plot")
        feature_window.geometry("400x300")
        feature_window.configure(bg=BACKGROUND_COLOR)
        
        tk.Label(feature_window, text="Select feature for dependency plot:", 
                font=("Helvetica", 12), bg=BACKGROUND_COLOR, fg=TEXT_COLOR).pack(pady=10)
        
        feature_var = tk.StringVar(value=X_processed.columns[0])
        
        feature_frame = tk.Frame(feature_window, bg=BACKGROUND_COLOR)
        feature_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        for feature in X_processed.columns:
            rb = tk.Radiobutton(feature_frame, text=feature, variable=feature_var, value=feature,
                              bg=BACKGROUND_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                              font=("Helvetica", 10))
            rb.pack(anchor=tk.W)
        
        def create_dependency_plot():
            feature_window.destroy()
            
            # Create dependency plot window
            dep_window = tk.Toplevel(root)
            dep_window.title("SHAP Dependency Plot")
            dep_window.geometry("1000x800")
            dep_window.configure(bg=BACKGROUND_COLOR)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create dependency plot
            feature_name = feature_var.get()
            
            if isinstance(shap_values, list):
                # For multi-class, use first class
                shap.dependence_plot(feature_name, shap_values[0], X_processed.values, 
                                   feature_names=X_processed.columns, show=False, ax=ax)
            else:
                shap.dependence_plot(feature_name, shap_values, X_processed.values, 
                                   feature_names=X_processed.columns, show=False, ax=ax)
            
            ax.set_title(f'SHAP Dependency Plot: {feature_name}', fontsize=16)
            plt.tight_layout()
            
            # Display plot
            canvas = FigureCanvasTkAgg(fig, master=dep_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Button to create plot
        create_btn = tk.Button(feature_window, text="Create Dependency Plot", 
                             command=create_dependency_plot,
                             bg=SUCCESS_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                             relief=tk.FLAT, padx=20, pady=10)
        create_btn.pack(pady=10)
        add_hover_effect(create_btn, hover_color=HOVER_COLOR, original_color=SUCCESS_COLOR)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create SHAP dependency plot: {str(e)}")

def show_data_type_info():
    """Show detailed information about data types in the dataset."""
    if df.empty:
        messagebox.showinfo("Data Type Info", "No data loaded!")
        return
    
    info_window = tk.Toplevel(root)
    info_window.title("Data Type Information")
    info_window.geometry("600x500")
    info_window.configure(bg=BACKGROUND_COLOR)
    
    # Create text widget
    text_frame = tk.Frame(info_window, bg=SECONDARY_COLOR)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    text_widget = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, 
                                          bg=SECONDARY_COLOR, fg=TEXT_COLOR,
                                          font=("Courier", 10))
    text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Generate data type information
    text_widget.insert(tk.END, "DATA TYPE INFORMATION\n\n", "header")
    
    # Column by column analysis
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].count()
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        
        text_widget.insert(tk.END, f"Column: {col}\n")
        text_widget.insert(tk.END, f"  Data Type: {dtype}\n")
        text_widget.insert(tk.END, f"  Non-null: {non_null}\n")
        text_widget.insert(tk.END, f"  Null: {null_count}\n")
        text_widget.insert(tk.END, f"  Unique: {unique_count}\n")
        
        # Show sample values for object/category types
        if dtype in ['object', 'category'] and unique_count <= 10:
            text_widget.insert(tk.END, f"  Unique values: {df[col].unique()}\n")
        
        text_widget.insert(tk.END, "\n")
    
    text_widget.config(state=tk.DISABLED)
    
def run_shap_dependency():
    """Run SHAP dependency plot."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Get data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Train a model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Create visualization window
        shap_window = tk.Toplevel(root)
        shap_window.title("SHAP Analysis - Dependency Plot")
        shap_window.geometry("1000x800")
        shap_window.configure(bg=BACKGROUND_COLOR)
        
        # Use first feature for dependency plot
        feature_name = X.columns[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.dependence_plot(feature_name, shap_values, X, show=False)
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=shap_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run SHAP dependency plot: {e}")

def compare_ensemble_methods():
    """Compare different ensemble methods."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Get data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Determine problem type
        if y.dtype == 'object' or y.nunique() < 10:
            problem_type = "Classification"
            # Encode target
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            problem_type = "Regression"
            y_encoded = y
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Define ensemble models
        if problem_type == "Classification":
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
                "LightGBM": LGBMClassifier(n_estimators=100, random_state=42),
                "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
                "Bagging": BaggingClassifier(n_estimators=100, random_state=42)
            }
        else:
            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
                "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
                "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
                "Bagging": BaggingRegressor(n_estimators=100, random_state=42)
            }
        
        # Train and evaluate models
        results = []
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if problem_type == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    results.append({"Model": name, "Accuracy": accuracy})
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    results.append({"Model": name, "MSE": mse, "R²": r2})
                    
            except Exception as e:
                print(f"Error with {name}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Display comparison results
        show_model_comparison(results_df, problem_type)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error in ensemble method comparison: {e}")

def run_feature_importance():
    """Run feature importance analysis."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Get data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Determine problem type
        if y.dtype == 'object' or y.nunique() < 10:
            problem_type = "Classification"
            # Encode target
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            problem_type = "Regression"
            y_encoded = y
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Train model
        model.fit(X, y_encoded)
        
        # Get feature importance
        importance = model.feature_importances_
        feature_names = X.columns
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Create visualization window
        importance_window = tk.Toplevel(root)
        importance_window.title("Feature Importance Analysis")
        importance_window.geometry("1000x800")
        importance_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot feature importance
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=ACCENT_COLOR)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance Analysis')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=importance_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run feature importance analysis: {e}")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def show_visualization_options():
    """Show visualization options with all plot types from the image."""
    popup = tk.Toplevel(root)
    popup.title("Visualization Options")
    popup.geometry("700x600")
    popup.configure(bg=BACKGROUND_COLOR)
    
    # Position the popup
    popup.geometry(f"+{root.winfo_x()+root.winfo_width()//2-350}+{root.winfo_y()+root.winfo_height()//2-300}")

    # Main container with stylish border
    main_frame = tk.Frame(popup, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    title_label = tk.Label(main_frame, text="Select Visualization Type", 
                         font=("Helvetica", 18, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    title_label.pack(pady=(10, 20))

    # Create a grid for visualization buttons
    grid_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    grid_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # Visualization options from the image
    visualizations = [
        ("Pie Chart", create_pie_chart),
        ("Bar Chart", create_bar_chart),
        ("Count Plot", create_count_plot),
        ("Scatter Plot", create_scatter_plot),
        ("Histogram", create_histogram),
        ("Dist Plot", create_dist_plot),
        ("KDE Plot", create_kde_plot),
        ("Line Plot", create_line_plot),
        ("Violin Plot", create_violin_plot),
        ("Area Plot", create_area_plot),
        ("Box Plot", create_box_plot),
        ("3D Plot", create_3d_plot)
    ]
    
    # Create buttons in a grid (3 columns)
    row, col = 0, 0
    for viz_name, viz_func in visualizations:
        btn = tk.Button(grid_frame, text=viz_name, command=viz_func,
                       bg=ACCENT_COLOR, fg="white", font=("Helvetica", 10, "bold"),
                       relief=tk.FLAT, padx=15, pady=10, bd=0, highlightthickness=0,
                       activebackground=HOVER_COLOR, width=15)
        btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        add_hover_effect(btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)
        
        col += 1
        if col >= 3:
            col = 0
            row += 1
    
    # Configure grid weights
    for i in range(3):
        grid_frame.columnconfigure(i, weight=1)
    for i in range(4):
        grid_frame.rowconfigure(i, weight=1)

    # Close button
    close_btn = tk.Button(main_frame, text="Close", command=popup.destroy,
                         bg=DANGER_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground="#B71C1C")
    close_btn.pack(pady=(20, 10))
    add_hover_effect(close_btn, hover_color="#B71C1C", original_color=DANGER_COLOR)

def create_pie_chart():
    """Create pie chart for categorical data."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) == 0:
            messagebox.showerror("Error", "No categorical columns found for pie chart!")
            return
        
        # Use first categorical column
        col = categorical_cols[0]
        value_counts = df[col].value_counts()
        
        # Limit to top 8 categories
        if len(value_counts) > 8:
            value_counts = value_counts.head(8)
            messagebox.showinfo("Info", "Showing top 8 categories for better visualization")
        
        # Create plot window
        plot_window = tk.Toplevel(root)
        plot_window.title("Pie Chart")
        plot_window.geometry("800x600")
        plot_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        
        # Style the text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(f'Pie Chart: {col}', fontsize=16)
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error creating pie chart: {e}")

def create_bar_chart():
    """Create bar chart for categorical data."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) == 0:
            messagebox.showerror("Error", "No categorical columns found for bar chart!")
            return
        
        # Use first categorical column
        col = categorical_cols[0]
        value_counts = df[col].value_counts()
        
        # Limit to top 10 categories
        if len(value_counts) > 10:
            value_counts = value_counts.head(10)
            messagebox.showinfo("Info", "Showing top 10 categories for better visualization")
        
        # Create plot window
        plot_window = tk.Toplevel(root)
        plot_window.title("Bar Chart")
        plot_window.geometry("800x600")
        plot_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        bars = ax.bar(value_counts.index, value_counts.values, color=ACCENT_COLOR, alpha=0.7)
        ax.set_title(f'Bar Chart: {col}', fontsize=16)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error creating bar chart: {e}")

def create_count_plot():
    """Create count plot for categorical data."""
    create_bar_chart()  # Count plot is similar to bar chart

def create_scatter_plot():
    """Create scatter plot for numeric data."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            messagebox.showerror("Error", "Need at least 2 numeric columns for scatter plot!")
            return
        
        # Create plot window
        plot_window = tk.Toplevel(root)
        plot_window.title("Scatter Plot")
        plot_window.geometry("800x600")
        plot_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot
        ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.7, color=ACCENT_COLOR)
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error creating scatter plot: {e}")

def create_histogram():
    """Create histogram for numeric data."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            messagebox.showerror("Error", "No numeric columns found for histogram!")
            return
        
        # Create plot window
        plot_window = tk.Toplevel(root)
        plot_window.title("Histogram")
        plot_window.geometry("800x600")
        plot_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        ax.hist(df[numeric_cols[0]], bins=20, color=ACCENT_COLOR, alpha=0.7, edgecolor='white')
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram: {numeric_cols[0]}', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error creating histogram: {e}")

def create_dist_plot():
    """Create distribution plot (similar to histogram with KDE)."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            messagebox.showerror("Error", "No numeric columns found for distribution plot!")
            return
        
        # Create plot window
        plot_window = tk.Toplevel(root)
        plot_window.title("Distribution Plot")
        plot_window.geometry("800x600")
        plot_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create distribution plot
        sns.histplot(df[numeric_cols[0]], kde=True, ax=ax, color=ACCENT_COLOR)
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution Plot: {numeric_cols[0]}', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error creating distribution plot: {e}")

def create_kde_plot():
    """Create KDE plot for numeric data."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            messagebox.showerror("Error", "No numeric columns found for KDE plot!")
            return
        
        # Create plot window
        plot_window = tk.Toplevel(root)
        plot_window.title("KDE Plot")
        plot_window.geometry("800x600")
        plot_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create KDE plot
        sns.kdeplot(df[numeric_cols[0]], ax=ax, color=ACCENT_COLOR, fill=True)
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel('Density')
        ax.set_title(f'KDE Plot: {numeric_cols[0]}', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error creating KDE plot: {e}")

def create_line_plot():
    """Create line plot for numeric data."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            messagebox.showerror("Error", "No numeric columns found for line plot!")
            return
        
        # Create plot window
        plot_window = tk.Toplevel(root)
        plot_window.title("Line Plot")
        plot_window.geometry("800x600")
        plot_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create line plot (using index as x-axis)
        ax.plot(df.index, df[numeric_cols[0]], color=ACCENT_COLOR, linewidth=2)
        ax.set_xlabel('Index')
        ax.set_ylabel(numeric_cols[0])
        ax.set_title(f'Line Plot: {numeric_cols[0]}', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error creating line plot: {e}")

def create_violin_plot():
    """Create violin plot for numeric data."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            messagebox.showerror("Error", "No numeric columns found for violin plot!")
            return
        
        # Create plot window
        plot_window = tk.Toplevel(root)
        plot_window.title("Violin Plot")
        plot_window.geometry("800x600")
        plot_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create violin plot
        sns.violinplot(y=df[numeric_cols[0]], ax=ax, color=ACCENT_COLOR)
        ax.set_ylabel(numeric_cols[0])
        ax.set_title(f'Violin Plot: {numeric_cols[0]}', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error creating violin plot: {e}")

def create_area_plot():
    """Create area plot for numeric data."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            messagebox.showerror("Error", "No numeric columns found for area plot!")
            return
        
        # Create plot window
        plot_window = tk.Toplevel(root)
        plot_window.title("Area Plot")
        plot_window.geometry("800x600")
        plot_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create area plot (using first few numeric columns)
        cols_to_plot = numeric_cols[:min(5, len(numeric_cols))]
        df[cols_to_plot].plot.area(ax=ax, alpha=0.7)
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title('Area Plot', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error creating area plot: {e}")

def create_box_plot():
    """Create box plot for numeric data."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            messagebox.showerror("Error", "No numeric columns found for box plot!")
            return
        
        # Create plot window
        plot_window = tk.Toplevel(root)
        plot_window.title("Box Plot")
        plot_window.geometry("800x600")
        plot_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create box plot
        df[numeric_cols].boxplot(ax=ax)
        ax.set_title('Box Plot of Numeric Columns', fontsize=16)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error creating box plot: {e}")

def create_3d_plot():
    """Create 3D scatter plot for numeric data."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 3:
            messagebox.showerror("Error", "Need at least 3 numeric columns for 3D plot!")
            return
        
        # Create plot window
        plot_window = tk.Toplevel(root)
        plot_window.title("3D Scatter Plot")
        plot_window.geometry("800x600")
        plot_window.configure(bg=BACKGROUND_COLOR)
        
        # Create figure with 3D axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create 3D scatter plot
        scatter = ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], df[numeric_cols[2]], 
                           c=df[numeric_cols[2]], cmap='viridis', alpha=0.7)
        
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_zlabel(numeric_cols[2])
        ax.set_title('3D Scatter Plot', fontsize=16)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        
        # Display plot
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error creating 3D plot: {e}")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Main application window
root = tk.Tk()
root.title("Machine Learning Analysis Dashboard")
root.geometry("1400x900")
root.configure(bg=BACKGROUND_COLOR)

# Apply modern theme
style = ttk.Style()
style.theme_use('clam')

# Configure styles
style.configure('TFrame', background=BACKGROUND_COLOR)
style.configure('TLabel', background=BACKGROUND_COLOR, foreground=TEXT_COLOR, font=('Helvetica', 10))
style.configure('TButton', background=ACCENT_COLOR, foreground='white', 
               font=('Helvetica', 10, 'bold'), borderwidth=0)
style.map('TButton', background=[('active', HOVER_COLOR)])
style.configure('TNotebook', background=BACKGROUND_COLOR)
style.configure('TNotebook.Tab', background=SECONDARY_COLOR, foreground=TEXT_COLOR,
               font=('Helvetica', 10, 'bold'), padding=[10, 5])
style.map('TNotebook.Tab', background=[('selected', ACCENT_COLOR)], foreground=[('selected', 'white')])

# Main container frame
main_container = tk.Frame(root, bg=BACKGROUND_COLOR)
main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Header frame with stylish border
header_frame = tk.Frame(main_container, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
header_frame.pack(fill=tk.X, pady=(0, 20))

# Title label
title_frame = tk.Frame(header_frame, bg=PRIMARY_COLOR)
title_frame.pack(fill=tk.X, pady=10)

main_label = tk.Label(title_frame, 
                     text="INSIGHT PREDICT: ML DATA ANALYZER",
                     font=("Helvetica", 24, "bold"), 
                     fg=TEXT_COLOR, bg=PRIMARY_COLOR)
main_label.pack(pady=(0, 5))

# Subtitle
sub_label = tk.Label(title_frame, 
                    text="An integrated ML solution for automated data analysis and model-driven insights",
                    font=("Helvetica", 12), 
                    fg=TEXT_COLOR, bg=PRIMARY_COLOR)
sub_label.pack()

# Summary and Workflow button
summary_frame = tk.Frame(header_frame, bg=PRIMARY_COLOR)
summary_frame.pack(fill=tk.X, pady=10)

summary_text = tk.Label(summary_frame,
                       text="InsightPredict is an automated ML analysis tool that preprocesses data, trains models, evaluates performance, and explains features to deliver fast and reliable insights",
                       font=("Helvetica", 10),
                       fg=TEXT_COLOR, bg=PRIMARY_COLOR, wraplength=1000)
summary_text.pack(side=tk.LEFT, padx=10)

workflow_btn = tk.Button(summary_frame, text="Workflow Guide", command=show_workflow,
                        bg=INFO_COLOR, fg="white", font=("Helvetica", 10, "bold"),
                        relief=tk.FLAT, padx=15, pady=5, bd=0, highlightthickness=0,
                        activebackground=HOVER_COLOR)
workflow_btn.pack(side=tk.RIGHT, padx=10)
add_hover_effect(workflow_btn, hover_color=HOVER_COLOR, original_color=INFO_COLOR)

# Separator
separator = ttk.Separator(header_frame, orient='horizontal')
separator.pack(fill=tk.X, pady=10)

# Content frame
content_frame = tk.Frame(main_container, bg=BACKGROUND_COLOR)
content_frame.pack(fill=tk.BOTH, expand=True)

# Left panel (buttons) with stylish border
left_panel = tk.Frame(content_frame, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

# Configure button style
btn_style = {
    'bg': ACCENT_COLOR,
    'fg': 'white',
    'font': ("Helvetica", 12, "bold"),
    'relief': tk.FLAT,
    'bd': 0,
    'highlightthickness': 0,
    'activebackground': HOVER_COLOR,
    'padx': 30,
    'pady': 15
}

# Load button
load_btn = tk.Button(left_panel, text="Load Dataset", command=load_file, **btn_style)
load_btn.pack(fill=tk.X, pady=(0, 15))
add_hover_effect(load_btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

# Prepare Data button
prepare_data_btn = tk.Button(left_panel, text="Prepare Data", command=show_prepare_data_interface,
                           **btn_style, state=tk.DISABLED)
prepare_data_btn.pack(fill=tk.X, pady=(0, 15))
add_hover_effect(prepare_data_btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

# Standardize Data button
standardize_btn = tk.Button(left_panel, text="Standardize Data", command=standardize_data,
                          **btn_style, state=tk.DISABLED)
standardize_btn.pack(fill=tk.X, pady=(0, 15))
add_hover_effect(standardize_btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

# Analyze Dataset button
analyze_dataset_btn = tk.Button(left_panel, text="Analyze Dataset", command=analyze_dataset,
                              **btn_style, state=tk.DISABLED)
analyze_dataset_btn.pack(fill=tk.X, pady=(0, 15))
add_hover_effect(analyze_dataset_btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

# Visualize Data button
visualize_btn = tk.Button(left_panel, text="Visualize Data", command=show_visualization_options,
                        **btn_style, state=tk.DISABLED)
visualize_btn.pack(fill=tk.X, pady=(0, 15))
add_hover_effect(visualize_btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

# Correlation Matrix button
corr_matrix_btn = tk.Button(left_panel, text="Correlation Matrix", command=show_correlation_matrix,
                          **btn_style, state=tk.DISABLED)
corr_matrix_btn.pack(fill=tk.X, pady=(0, 15))
add_hover_effect(corr_matrix_btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

# Advanced Analysis button
advanced_analysis_btn = tk.Button(left_panel, text="Advanced Analysis", command=show_advanced_analysis,
                                **btn_style, state=tk.DISABLED)
advanced_analysis_btn.pack(fill=tk.X, pady=(0, 15))
add_hover_effect(advanced_analysis_btn, hover_color=HOVER_COLOR, original_color=ACCENT_COLOR)

# Right panel (data preview and info) with stylish border
right_panel = tk.Frame(content_frame, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Data preview label
preview_label = tk.Label(right_panel, text="Dataset Overview", 
                        font=("Helvetica", 14, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
preview_label.pack(pady=(10, 10), anchor=tk.W, padx=10)

# Data preview text widget with scrollbar
preview_frame = tk.Frame(right_panel, bg=PRIMARY_COLOR)
preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

data_preview = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD, 
                                       bg=SECONDARY_COLOR, fg=TEXT_COLOR, 
                                       insertbackground=TEXT_COLOR, 
                                       font=("Courier", 10),
                                       padx=10, pady=10)
data_preview.pack(fill=tk.BOTH, expand=True)

# Configure text tags for styling
data_preview.tag_config("header", foreground=TEXT_COLOR, font=("Helvetica", 12, "bold"))
data_preview.tag_config("info", foreground=TEXT_COLOR, font=("Helvetica", 10))
data_preview.tag_config("data", foreground=TEXT_COLOR, font=("Courier", 10))
data_preview.tag_config("warning", foreground=WARNING_COLOR, font=("Helvetica", 10))

data_preview.insert(tk.END, "No data loaded. Please click 'Load Dataset' to begin.", "info")
data_preview.config(state=tk.DISABLED)

# Footer with stylish border
footer_frame = tk.Frame(main_container, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE, highlightbackground=BORDER_COLOR, highlightthickness=2)
footer_frame.pack(fill=tk.X, pady=(20, 0))

# Exit button with modern style
exit_btn = tk.Button(footer_frame, text="Exit", command=root.quit,
                    bg=DANGER_COLOR, fg="white", font=("Helvetica", 12, "bold"),
                    relief=tk.FLAT, padx=30, pady=10, bd=0, highlightthickness=0,
                    activebackground="#B71C1C")
exit_btn.pack(pady=10)
add_hover_effect(exit_btn, hover_color="#B71C1C", original_color=DANGER_COLOR)

# Initialize empty DataFrame
df = pd.DataFrame()
original_df = pd.DataFrame()
X = pd.DataFrame()
y = pd.Series()

# Start the application
root.mainloop()