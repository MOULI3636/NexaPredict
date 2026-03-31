import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np
from tkinter import scrolledtext
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import shap
from pandastable import Table
import matplotlib.ticker as ticker
from matplotlib import rcParams

# Modern luxury color scheme
BACKGROUND_COLOR = "#121212"
PRIMARY_COLOR = "#1E1E1E"
SECONDARY_COLOR = "#252525"
ACCENT_COLOR = "#7E57C2"
HOVER_COLOR = "#9575CD"
TEXT_COLOR = "#E0E0E0"
DANGER_COLOR = "#D32F2F"
SUCCESS_COLOR = "#388E3C"
INFO_COLOR = "#1976D2"
HIGHLIGHT_COLOR = "#BB86FC"

# Configure matplotlib style
plt.style.use('dark_background')
rcParams['axes.titlecolor'] = TEXT_COLOR
rcParams['axes.labelcolor'] = TEXT_COLOR
rcParams['xtick.color'] = TEXT_COLOR
rcParams['ytick.color'] = TEXT_COLOR
rcParams['figure.facecolor'] = PRIMARY_COLOR
rcParams['axes.facecolor'] = PRIMARY_COLOR
rcParams['grid.color'] = SECONDARY_COLOR

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
    analyze_btn.config(state=tk.NORMAL)
    corr_matrix_btn.config(state=tk.NORMAL)
    boosting_btn.config(state=tk.NORMAL)
    advanced_btn.config(state=tk.NORMAL)

def show_regression_options():
    """Show regression options popup."""
    popup = tk.Toplevel(root)
    popup.title("Regression Options")
    popup.geometry("500x400")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)
    
    # Make the popup appear in the center of the main window
    popup.geometry(f"+{root.winfo_x()+root.winfo_width()//2-250}+{root.winfo_y()+root.winfo_height()//2-200}")

    # Create a stylish frame for the popup content
    content_frame = tk.Frame(popup, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
    content_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

    label = tk.Label(content_frame, text="Select Analysis Type", 
                    font=("Helvetica", 16, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
    label.pack(pady=(20, 30))

    # Button frame
    button_frame = tk.Frame(content_frame, bg=PRIMARY_COLOR)
    button_frame.pack(pady=10)

    linear_btn = tk.Button(button_frame, text="Linear Regression", 
                         command=lambda: [popup.destroy(), show_linear_regression_interface()], 
                         bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=30, pady=12, bd=0, highlightthickness=0,
                         activebackground=HOVER_COLOR)
    linear_btn.pack(pady=10, fill=tk.X)
    add_hover_effect(linear_btn)

    logistic_btn = tk.Button(button_frame, text="Logistic Regression", 
                           command=lambda: [popup.destroy(), show_classification_interface()], 
                           bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                           relief=tk.FLAT, padx=30, pady=12, bd=0, highlightthickness=0,
                           activebackground=HOVER_COLOR)
    logistic_btn.pack(pady=10, fill=tk.X)
    add_hover_effect(logistic_btn)

    close_btn = tk.Button(content_frame, text="Cancel", command=popup.destroy,
                         bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=("Helvetica", 10),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground=SECONDARY_COLOR)
    close_btn.pack(pady=(20, 10))
    add_hover_effect(close_btn, hover_color=PRIMARY_COLOR)

def add_hover_effect(button, hover_color=HOVER_COLOR):
    """Function to add hover effect to a button."""
    def on_enter(e):
        button.config(bg=hover_color)

    def on_leave(e):
        button.config(bg=button.cget('bg'))  # Return to original color

    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

def show_classification_interface():
    """Display interface for classification analysis."""
    try:
        popup = tk.Toplevel(root)
        popup.title("Classification Analysis")
        popup.geometry("1200x900")
        popup.configure(bg=BACKGROUND_COLOR)
        
        # Position the popup relative to main window
        popup.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

        # Main container frame with stylish border
        main_frame = tk.Frame(popup, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        title_label = tk.Label(main_frame, text="Classification Analysis Results", 
                             font=("Helvetica", 18, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
        title_label.pack(pady=(10, 20))

        global X, y
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Splitting data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Generate metrics
        report = classification_report(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Results frame
        results_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Text results in a stylish frame
        text_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        result_text = tk.Text(text_frame, wrap=tk.WORD, height=15, bg=SECONDARY_COLOR, 
                            fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Courier", 10),
                            padx=10, pady=10)
        scrollbar = ttk.Scrollbar(text_frame, command=result_text.yview)
        result_text.config(yscrollcommand=scrollbar.set)
        
        result_text.insert(tk.END, f"Classification Report:\n{report}\n")
        result_text.insert(tk.END, f"\nAccuracy: {acc:.2f}\n")
        result_text.config(state=tk.DISABLED)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Confusion matrix plot in a stylish frame
        plot_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax, 
                   annot_kws={"size": 12, "color": "white"})
        ax.set_title("Confusion Matrix", fontsize=14, pad=20)
        ax.set_xlabel("Predicted Labels", fontsize=12)
        ax.set_ylabel("True Labels", fontsize=12)
        
        # Add border to the plot
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color(HIGHLIGHT_COLOR)
            spine.set_linewidth(2)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Feature importance (for models that support it)
        if hasattr(model, 'coef_'):
            feature_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
            feature_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
            
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            coef = model.coef_[0]
            features = X.columns
            sorted_idx = np.argsort(np.abs(coef))
            
            ax2.barh(range(len(sorted_idx)), np.abs(coef[sorted_idx]), 
                   color=ACCENT_COLOR, alpha=0.8)
            ax2.set_yticks(range(len(sorted_idx)))
            ax2.set_yticklabels([features[i] for i in sorted_idx])
            ax2.set_title("Feature Importance (Absolute Coefficients)", fontsize=14)
            ax2.set_xlabel("Absolute Coefficient Value", fontsize=12)
            
            # Add border to the plot
            for _, spine in ax2.spines.items():
                spine.set_visible(True)
                spine.set_color(HIGHLIGHT_COLOR)
                spine.set_linewidth(2)
            
            canvas2 = FigureCanvasTkAgg(fig2, master=feature_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Close button with modern style
        close_btn = tk.Button(main_frame, text="Close", command=popup.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                            relief=tk.FLAT, padx=30, pady=8, bd=0, highlightthickness=0,
                            activebackground=HOVER_COLOR)
        close_btn.pack(pady=(10, 20))
        add_hover_effect(close_btn)

    except Exception as e:
        messagebox.showerror("Error", f"Error during analysis: {e}")

def show_linear_regression_interface():
    """Display interface for linear regression analysis."""
    try:
        popup = tk.Toplevel(root)
        popup.title("Regression Analysis")
        popup.geometry("1300x1000")
        popup.configure(bg=BACKGROUND_COLOR)
        
        # Position the popup relative to main window
        popup.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

        # Main container frame with stylish border
        main_frame = tk.Frame(popup, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        title_label = tk.Label(main_frame, text="Regression Analysis Results", 
                             font=("Helvetica", 18, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
        title_label.pack(pady=(10, 20))

        global X, y
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        coefficients = model.coef_
        intercept = model.intercept_

        # Results frame
        results_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Text results in a stylish frame
        text_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))

        result_text = tk.Text(text_frame, wrap=tk.WORD, height=15, bg=SECONDARY_COLOR, 
                            fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Courier", 10),
                            padx=10, pady=10)
        scrollbar = ttk.Scrollbar(text_frame, command=result_text.yview)
        result_text.config(yscrollcommand=scrollbar.set)
        
        result_text.insert(tk.END, "Regression Model Summary:\n\n")
        result_text.insert(tk.END, f"Formula: y = {intercept:.4f} ")
        for i, coef in enumerate(coefficients, 1):
            result_text.insert(tk.END, f"+ {coef:.4f} * x{i} ")
        result_text.insert(tk.END, "\n\n")
        result_text.insert(tk.END, f"Mean Squared Error (MSE): {mse:.4f}\n")
        result_text.insert(tk.END, f"R² Score: {r2:.4f}\n\n")
        result_text.insert(tk.END, "Feature Coefficients:\n")
        for i, coef in enumerate(coefficients, 1):
            result_text.insert(tk.END, f"  x{i}: {coef:.6f}\n")
        result_text.config(state=tk.DISABLED)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Plot frame with stylish border
        plot_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Plotting True vs Predicted values
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_test, y_pred, color=ACCENT_COLOR, edgecolor='k', alpha=0.7)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                color='red', linestyle='--', linewidth=2)
        ax.set_title("True vs Predicted Values", fontsize=14)
        ax.set_xlabel("True Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add border to the plot
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color(HIGHLIGHT_COLOR)
            spine.set_linewidth(2)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Residual plot
        residual_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
        residual_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, color=ACCENT_COLOR, alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_title("Residual Plot", fontsize=14)
        ax2.set_xlabel("Predicted Values", fontsize=12)
        ax2.set_ylabel("Residuals", fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Add border to the plot
        for _, spine in ax2.spines.items():
            spine.set_visible(True)
            spine.set_color(HIGHLIGHT_COLOR)
            spine.set_linewidth(2)
        
        canvas2 = FigureCanvasTkAgg(fig2, master=residual_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Close button with modern style
        close_btn = tk.Button(main_frame, text="Close", command=popup.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                            relief=tk.FLAT, padx=30, pady=8, bd=0, highlightthickness=0,
                            activebackground=HOVER_COLOR)
        close_btn.pack(pady=(10, 20))
        add_hover_effect(close_btn)

    except Exception as e:
        messagebox.showerror("Error", f"Error during analysis: {e}")

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
        main_frame = tk.Frame(corr_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(main_frame, text="Feature Correlation Matrix", 
                             font=("Helvetica", 18, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
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
        sns.heatmap(corr_matrix, annot=True, cmap='Purples', center=0, ax=ax,
                   annot_kws={"size": 10, "color": "white"}, fmt=".2f", 
                   linewidths=.5, cbar_kws={"shrink": 0.8})
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Add border to the plot
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color(HIGHLIGHT_COLOR)
            spine.set_linewidth(2)

        # Embed in Tkinter with stylish frame
        plot_frame = tk.Frame(main_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Close button with modern style
        close_btn = tk.Button(main_frame, text="Close", command=corr_window.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                            relief=tk.FLAT, padx=30, pady=8, bd=0, highlightthickness=0,
                            activebackground=HOVER_COLOR)
        close_btn.pack(pady=(10, 20))
        add_hover_effect(close_btn)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create correlation matrix: {e}")

def show_boosting_options():
    """Show boosting algorithm options popup with improved styling."""
    popup = tk.Toplevel(root)
    popup.title("Boosting Algorithm Options")
    popup.geometry("500x400")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)
    
    # Position the popup
    popup.geometry(f"+{root.winfo_x()+root.winfo_width()//2-250}+{root.winfo_y()+root.winfo_height()//2-200}")

    # Stylish content frame
    content_frame = tk.Frame(popup, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
    content_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

    label = tk.Label(content_frame, text="Select Boosting Algorithm", 
                    font=("Helvetica", 16, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
    label.pack(pady=(20, 30))

    # Button frame
    button_frame = tk.Frame(content_frame, bg=PRIMARY_COLOR)
    button_frame.pack(pady=10)

    xgb_btn = tk.Button(button_frame, text="XGBoost", 
                       command=lambda: [popup.destroy(), run_boosting_algorithm("XGBoost")], 
                       bg=INFO_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                       relief=tk.FLAT, padx=30, pady=12, bd=0, highlightthickness=0,
                       activebackground=HOVER_COLOR)
    xgb_btn.pack(pady=10, fill=tk.X)
    add_hover_effect(xgb_btn)

    cat_btn = tk.Button(button_frame, text="CatBoost", 
                       command=lambda: [popup.destroy(), run_boosting_algorithm("CatBoost")], 
                       bg=INFO_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                       relief=tk.FLAT, padx=30, pady=12, bd=0, highlightthickness=0,
                       activebackground=HOVER_COLOR)
    cat_btn.pack(pady=10, fill=tk.X)
    add_hover_effect(cat_btn)

    # Close button
    close_btn = tk.Button(content_frame, text="Cancel", command=popup.destroy,
                         bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=("Helvetica", 10),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground=SECONDARY_COLOR)
    close_btn.pack(pady=(20, 10))
    add_hover_effect(close_btn, hover_color=PRIMARY_COLOR)

def run_boosting_algorithm(algorithm):
    """Run the selected boosting algorithm and display results with improved styling."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Determine if classification or regression
        problem_type = "classification" if df.iloc[:, -1].nunique() < 10 else "regression"
        
        # Split data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize model
        if algorithm == "XGBoost":
            if problem_type == "classification":
                model = XGBClassifier(random_state=42, eval_metric='mlogloss')
            else:
                model = XGBRegressor(random_state=42)
        else:  # CatBoost
            if problem_type == "classification":
                model = CatBoostClassifier(random_state=42, silent=True)
            else:
                model = CatBoostRegressor(random_state=42, silent=True)
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Create results window with improved styling
        results_window = tk.Toplevel(root)
        results_window.title(f"{algorithm} Results")
        results_window.geometry("1200x900")
        results_window.configure(bg=BACKGROUND_COLOR)
        
        # Position the window
        results_window.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

        # Main container with stylish border
        main_frame = tk.Frame(results_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text=f"{algorithm} Results ({problem_type.capitalize()})", 
                             font=("Helvetica", 18, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
        title_label.pack(pady=(10, 20))
        
        # Results frame
        results_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Text results in stylish frame
        text_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        result_text = tk.Text(text_frame, wrap=tk.WORD, height=12, bg=SECONDARY_COLOR, 
                            fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Courier", 10),
                            padx=10, pady=10)
        scrollbar = ttk.Scrollbar(text_frame, command=result_text.yview)
        result_text.config(yscrollcommand=scrollbar.set)
        
        if problem_type == "classification":
            report = classification_report(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            result_text.insert(tk.END, f"Classification Report:\n{report}\n")
            result_text.insert(tk.END, f"\nAccuracy: {acc:.4f}\n")
            
            # Confusion matrix plot in stylish frame
            plot_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
            plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax1,
                       annot_kws={"size": 12, "color": "white"})
            ax1.set_title("Confusion Matrix", fontsize=14)
            ax1.set_xlabel("Predicted Labels", fontsize=12)
            ax1.set_ylabel("True Labels", fontsize=12)
            
            # Add border to the plot
            for _, spine in ax1.spines.items():
                spine.set_visible(True)
                spine.set_color(HIGHLIGHT_COLOR)
                spine.set_linewidth(2)
            
            canvas1 = FigureCanvasTkAgg(fig1, master=plot_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            result_text.insert(tk.END, "Regression Metrics:\n\n")
            result_text.insert(tk.END, f"Mean Squared Error (MSE): {mse:.6f}\n")
            result_text.insert(tk.END, f"R² Score: {r2:.4f}\n")
            
            # True vs Predicted plot in stylish frame
            plot_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
            plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            ax1.scatter(y_test, y_pred, color=ACCENT_COLOR, edgecolor='k', alpha=0.7)
            ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                    color='red', linestyle='--', linewidth=2)
            ax1.set_title("True vs Predicted Values", fontsize=14)
            ax1.set_xlabel("True Values", fontsize=12)
            ax1.set_ylabel("Predicted Values", fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            # Add border to the plot
            for _, spine in ax1.spines.items():
                spine.set_visible(True)
                spine.set_color(HIGHLIGHT_COLOR)
                spine.set_linewidth(2)
            
            canvas1 = FigureCanvasTkAgg(fig1, master=plot_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        result_text.config(state=tk.DISABLED)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Feature importance plot in stylish frame
        feature_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
        feature_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        if algorithm == "XGBoost":
            importances = model.feature_importances_
            features = X.columns
            sorted_idx = np.argsort(importances)
            ax2.barh(range(len(sorted_idx)), importances[sorted_idx], color=ACCENT_COLOR)
            ax2.set_yticks(range(len(sorted_idx)))
            ax2.set_yticklabels([features[i] for i in sorted_idx])
        else:  # CatBoost
            importances = model.get_feature_importance()
            features = X.columns
            sorted_idx = np.argsort(importances)
            ax2.barh(range(len(sorted_idx)), importances[sorted_idx], color=ACCENT_COLOR)
            ax2.set_yticks(range(len(sorted_idx)))
            ax2.set_yticklabels([features[i] for i in sorted_idx])
        
        ax2.set_title("Feature Importance", fontsize=14)
        ax2.set_xlabel("Importance Score", fontsize=12)
        
        # Add border to the plot
        for _, spine in ax2.spines.items():
            spine.set_visible(True)
            spine.set_color(HIGHLIGHT_COLOR)
            spine.set_linewidth(2)
        
        plot_frame2 = tk.Frame(feature_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
        plot_frame2.pack(fill=tk.BOTH, expand=True)
        
        canvas2 = FigureCanvasTkAgg(fig2, master=plot_frame2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Close button with modern style
        close_btn = tk.Button(main_frame, text="Close", command=results_window.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                            relief=tk.FLAT, padx=30, pady=8, bd=0, highlightthickness=0,
                            activebackground=HOVER_COLOR)
        close_btn.pack(pady=(10, 20))
        add_hover_effect(close_btn)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error during {algorithm} analysis: {e}")

def show_advanced_analysis_options():
    """Show advanced analysis options popup with improved styling."""
    popup = tk.Toplevel(root)
    popup.title("Advanced Analysis Options")
    popup.geometry("500x400")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)
    
    # Position the popup
    popup.geometry(f"+{root.winfo_x()+root.winfo_width()//2-250}+{root.winfo_y()+root.winfo_height()//2-200}")

    # Stylish content frame
    content_frame = tk.Frame(popup, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
    content_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

    label = tk.Label(content_frame, text="Select Advanced Analysis", 
                    font=("Helvetica", 16, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
    label.pack(pady=(20, 30))

    # Button frame
    button_frame = tk.Frame(content_frame, bg=PRIMARY_COLOR)
    button_frame.pack(pady=10)

    ensemble_btn = tk.Button(button_frame, text="Ensemble Methods", 
                           command=lambda: [popup.destroy(), show_ensemble_options()], 
                           bg=INFO_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                           relief=tk.FLAT, padx=30, pady=12, bd=0, highlightthickness=0,
                           activebackground=HOVER_COLOR)
    ensemble_btn.pack(pady=10, fill=tk.X)
    add_hover_effect(ensemble_btn)

    pca_btn = tk.Button(button_frame, text="PCA Analysis", 
                       command=lambda: [popup.destroy(), run_pca_analysis()], 
                       bg=INFO_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                       relief=tk.FLAT, padx=30, pady=12, bd=0, highlightthickness=0,
                       activebackground=HOVER_COLOR)
    pca_btn.pack(pady=10, fill=tk.X)
    add_hover_effect(pca_btn)

    shap_btn = tk.Button(button_frame, text="SHAP Analysis", 
                        command=lambda: [popup.destroy(), show_shap_options()], 
                        bg=INFO_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                        relief=tk.FLAT, padx=30, pady=12, bd=0, highlightthickness=0,
                        activebackground=HOVER_COLOR)
    shap_btn.pack(pady=10, fill=tk.X)
    add_hover_effect(shap_btn)

    # Close button
    close_btn = tk.Button(content_frame, text="Cancel", command=popup.destroy,
                         bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=("Helvetica", 10),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground=SECONDARY_COLOR)
    close_btn.pack(pady=(20, 10))
    add_hover_effect(close_btn, hover_color=PRIMARY_COLOR)

def show_ensemble_options():
    """Show ensemble method options popup with improved styling."""
    popup = tk.Toplevel(root)
    popup.title("Ensemble Method Options")
    popup.geometry("500x400")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)
    
    # Position the popup
    popup.geometry(f"+{root.winfo_x()+root.winfo_width()//2-250}+{root.winfo_y()+root.winfo_height()//2-200}")

    # Stylish content frame
    content_frame = tk.Frame(popup, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
    content_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

    label = tk.Label(content_frame, text="Select Ensemble Method", 
                    font=("Helvetica", 16, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
    label.pack(pady=(20, 30))

    # Button frame
    button_frame = tk.Frame(content_frame, bg=PRIMARY_COLOR)
    button_frame.pack(pady=10)

    voting_btn = tk.Button(button_frame, text="Voting Ensemble", 
                          command=lambda: [popup.destroy(), run_ensemble_analysis("Voting")], 
                          bg=INFO_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                          relief=tk.FLAT, padx=30, pady=12, bd=0, highlightthickness=0,
                          activebackground=HOVER_COLOR)
    voting_btn.pack(pady=10, fill=tk.X)
    add_hover_effect(voting_btn)

    rf_btn = tk.Button(button_frame, text="Random Forest", 
                      command=lambda: [popup.destroy(), run_ensemble_analysis("RandomForest")], 
                      bg=INFO_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                      relief=tk.FLAT, padx=30, pady=12, bd=0, highlightthickness=0,
                      activebackground=HOVER_COLOR)
    rf_btn.pack(pady=10, fill=tk.X)
    add_hover_effect(rf_btn)

    # Close button
    close_btn = tk.Button(content_frame, text="Cancel", command=popup.destroy,
                         bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=("Helvetica", 10),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground=SECONDARY_COLOR)
    close_btn.pack(pady=(20, 10))
    add_hover_effect(close_btn, hover_color=PRIMARY_COLOR)

def run_ensemble_analysis(method):
    """Run the selected ensemble method and display results with improved styling."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Determine if classification or regression
        problem_type = "classification" if df.iloc[:, -1].nunique() < 10 else "regression"
        
        # Split data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize model
        if method == "Voting":
            if problem_type == "classification":
                model = VotingClassifier(estimators=[
                    ('lr', LogisticRegression(max_iter=1000)),
                    ('rf', RandomForestClassifier(random_state=42)),
                    ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss'))
                ], voting='soft')
            else:
                model = VotingRegressor(estimators=[
                    ('lr', LinearRegression()),
                    ('rf', RandomForestRegressor(random_state=42)),
                    ('xgb', XGBRegressor(random_state=42))
                ])
        else:  # RandomForest
            if problem_type == "classification":
                model = RandomForestClassifier(random_state=42)
            else:
                model = RandomForestRegressor(random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Create results window with improved styling
        results_window = tk.Toplevel(root)
        results_window.title(f"{method} Ensemble Results")
        results_window.geometry("1200x900")
        results_window.configure(bg=BACKGROUND_COLOR)
        
        # Position the window
        results_window.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

        # Main container with stylish border
        main_frame = tk.Frame(results_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text=f"{method} Ensemble Results ({problem_type.capitalize()})", 
                             font=("Helvetica", 18, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
        title_label.pack(pady=(10, 20))
        
        # Results frame
        results_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Text results in stylish frame
        text_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        result_text = tk.Text(text_frame, wrap=tk.WORD, height=12, bg=SECONDARY_COLOR, 
                            fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Courier", 10),
                            padx=10, pady=10)
        scrollbar = ttk.Scrollbar(text_frame, command=result_text.yview)
        result_text.config(yscrollcommand=scrollbar.set)
        
        if problem_type == "classification":
            report = classification_report(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            result_text.insert(tk.END, f"Classification Report:\n{report}\n")
            result_text.insert(tk.END, f"\nAccuracy: {acc:.4f}\n")
            
            # Confusion matrix plot in stylish frame
            plot_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
            plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax1,
                       annot_kws={"size": 12, "color": "white"})
            ax1.set_title("Confusion Matrix", fontsize=14)
            ax1.set_xlabel("Predicted Labels", fontsize=12)
            ax1.set_ylabel("True Labels", fontsize=12)
            
            # Add border to the plot
            for _, spine in ax1.spines.items():
                spine.set_visible(True)
                spine.set_color(HIGHLIGHT_COLOR)
                spine.set_linewidth(2)
            
            canvas1 = FigureCanvasTkAgg(fig1, master=plot_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            result_text.insert(tk.END, "Regression Metrics:\n\n")
            result_text.insert(tk.END, f"Mean Squared Error (MSE): {mse:.6f}\n")
            result_text.insert(tk.END, f"R² Score: {r2:.4f}\n")
            
            # True vs Predicted plot in stylish frame
            plot_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
            plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            ax1.scatter(y_test, y_pred, color=ACCENT_COLOR, edgecolor='k', alpha=0.7)
            ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                    color='red', linestyle='--', linewidth=2)
            ax1.set_title("True vs Predicted Values", fontsize=14)
            ax1.set_xlabel("True Values", fontsize=12)
            ax1.set_ylabel("Predicted Values", fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            # Add border to the plot
            for _, spine in ax1.spines.items():
                spine.set_visible(True)
                spine.set_color(HIGHLIGHT_COLOR)
                spine.set_linewidth(2)
            
            canvas1 = FigureCanvasTkAgg(fig1, master=plot_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        result_text.config(state=tk.DISABLED)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Feature importance plot (only for Random Forest)
        if method == "RandomForest":
            feature_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
            feature_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
            
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            importances = model.feature_importances_
            features = X.columns
            sorted_idx = np.argsort(importances)
            ax2.barh(range(len(sorted_idx)), importances[sorted_idx], color=ACCENT_COLOR)
            ax2.set_yticks(range(len(sorted_idx)))
            ax2.set_yticklabels([features[i] for i in sorted_idx])
            ax2.set_title("Feature Importance", fontsize=14)
            ax2.set_xlabel("Importance Score", fontsize=12)
            
            # Add border to the plot
            for _, spine in ax2.spines.items():
                spine.set_visible(True)
                spine.set_color(HIGHLIGHT_COLOR)
                spine.set_linewidth(2)
            
            plot_frame2 = tk.Frame(feature_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
            plot_frame2.pack(fill=tk.BOTH, expand=True)
            
            canvas2 = FigureCanvasTkAgg(fig2, master=plot_frame2)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Close button with modern style
        close_btn = tk.Button(main_frame, text="Close", command=results_window.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                            relief=tk.FLAT, padx=30, pady=8, bd=0, highlightthickness=0,
                            activebackground=HOVER_COLOR)
        close_btn.pack(pady=(10, 20))
        add_hover_effect(close_btn)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error during {method} ensemble analysis: {e}")

def run_pca_analysis():
    """Run PCA analysis and display results with improved styling."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            messagebox.showerror("Error", "No numeric columns found for PCA!")
            return
            
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Run PCA
        pca = PCA()
        pca.fit(scaled_data)
        
        # Create results window with improved styling
        results_window = tk.Toplevel(root)
        results_window.title("PCA Analysis Results")
        results_window.geometry("1200x900")
        results_window.configure(bg=BACKGROUND_COLOR)
        
        # Position the window
        results_window.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

        # Main container with stylish border
        main_frame = tk.Frame(results_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="Principal Component Analysis (PCA)", 
                             font=("Helvetica", 18, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
        title_label.pack(pady=(10, 20))
        
        # Results frame
        results_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Text results in stylish frame
        text_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        result_text = tk.Text(text_frame, wrap=tk.WORD, height=12, bg=SECONDARY_COLOR, 
                            fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Courier", 10),
                            padx=10, pady=10)
        scrollbar = ttk.Scrollbar(text_frame, command=result_text.yview)
        result_text.config(yscrollcommand=scrollbar.set)
        
        result_text.insert(tk.END, "PCA Explained Variance Ratio:\n")
        for i, ratio in enumerate(pca.explained_variance_ratio_, 1):
            result_text.insert(tk.END, f"PC{i}: {ratio:.4f} ({ratio*100:.1f}%)\n")
        
        result_text.insert(tk.END, f"\nTotal Explained Variance: {sum(pca.explained_variance_ratio_):.4f}\n")
        result_text.insert(tk.END, "\nPCA Components (First 5):\n")
        components_df = pd.DataFrame(pca.components_, columns=numeric_df.columns)
        result_text.insert(tk.END, components_df.head().to_string())
        
        result_text.config(state=tk.DISABLED)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scree plot in stylish frame
        plot_frame = tk.Frame(results_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.plot(range(1, len(pca.explained_variance_ratio_)+1), 
                pca.explained_variance_ratio_, 'o-', color=ACCENT_COLOR, markersize=8)
        ax1.set_title("Scree Plot", fontsize=14)
        ax1.set_xlabel("Principal Component", fontsize=12)
        ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Add border to the plot
        for _, spine in ax1.spines.items():
            spine.set_visible(True)
            spine.set_color(HIGHLIGHT_COLOR)
            spine.set_linewidth(2)
        
        canvas1 = FigureCanvasTkAgg(fig1, master=plot_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a new frame for the second plot
        plot_frame2 = tk.Frame(main_frame, bg=PRIMARY_COLOR)
        plot_frame2.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        # PCA biplot (first two components)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        transformed_data = pca.transform(scaled_data)
        
        # Scatter plot of first two components
        scatter = ax2.scatter(transformed_data[:, 0], transformed_data[:, 1], 
                   color=ACCENT_COLOR, alpha=0.7)
        ax2.set_title("PCA Biplot (First Two Components)", fontsize=14)
        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
        ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
        
        # Add feature vectors
        for i, feature in enumerate(numeric_df.columns):
            ax2.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
                     color='r', alpha=0.7, head_width=0.05)
            ax2.text(pca.components_[0, i]*1.15, pca.components_[1, i]*1.15, 
                    feature, color='r', ha='center', va='center', fontsize=10)
        
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Add border to the plot
        for _, spine in ax2.spines.items():
            spine.set_visible(True)
            spine.set_color(HIGHLIGHT_COLOR)
            spine.set_linewidth(2)
        
        # Embed in stylish frame
        plot_frame3 = tk.Frame(plot_frame2, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
        plot_frame3.pack(fill=tk.BOTH, expand=True)
        
        canvas2 = FigureCanvasTkAgg(fig2, master=plot_frame3)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Close button with modern style
        close_btn = tk.Button(main_frame, text="Close", command=results_window.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                            relief=tk.FLAT, padx=30, pady=8, bd=0, highlightthickness=0,
                            activebackground=HOVER_COLOR)
        close_btn.pack(pady=(10, 20))
        add_hover_effect(close_btn)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error during PCA analysis: {e}")

def show_shap_options():
    """Show SHAP analysis options popup with improved styling."""
    popup = tk.Toplevel(root)
    popup.title("SHAP Analysis Options")
    popup.geometry("500x400")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)
    
    # Position the popup
    popup.geometry(f"+{root.winfo_x()+root.winfo_width()//2-250}+{root.winfo_y()+root.winfo_height()//2-200}")

    # Stylish content frame
    content_frame = tk.Frame(popup, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
    content_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

    label = tk.Label(content_frame, text="Select SHAP Visualization", 
                    font=("Helvetica", 16, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
    label.pack(pady=(20, 30))

    # Button frame
    button_frame = tk.Frame(content_frame, bg=PRIMARY_COLOR)
    button_frame.pack(pady=10)

    summary_btn = tk.Button(button_frame, text="Summary Plot", 
                          command=lambda: [popup.destroy(), run_shap_analysis("summary")], 
                          bg=INFO_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                          relief=tk.FLAT, padx=30, pady=12, bd=0, highlightthickness=0,
                          activebackground=HOVER_COLOR)
    summary_btn.pack(pady=10, fill=tk.X)
    add_hover_effect(summary_btn)

    dependence_btn = tk.Button(button_frame, text="Dependence Plot", 
                             command=lambda: [popup.destroy(), run_shap_analysis("dependence")], 
                             bg=INFO_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                             relief=tk.FLAT, padx=30, pady=12, bd=0, highlightthickness=0,
                             activebackground=HOVER_COLOR)
    dependence_btn.pack(pady=10, fill=tk.X)
    add_hover_effect(dependence_btn)

    # Close button
    close_btn = tk.Button(content_frame, text="Cancel", command=popup.destroy,
                         bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=("Helvetica", 10),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground=SECONDARY_COLOR)
    close_btn.pack(pady=(20, 10))
    add_hover_effect(close_btn, hover_color=PRIMARY_COLOR)

def run_shap_analysis(plot_type):
    """Run SHAP analysis and display the selected plot with improved styling."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Determine if classification or regression
        problem_type = "classification" if df.iloc[:, -1].nunique() < 10 else "regression"
        
        # Split data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a model (using XGBoost for SHAP values)
        if problem_type == "classification":
            model = XGBClassifier(random_state=42, eval_metric='mlogloss')
        else:
            model = XGBRegressor(random_state=42)
        
        model.fit(X_train, y_train)
        
        # Calculate SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        
        # Create results window with improved styling
        results_window = tk.Toplevel(root)
        results_window.title(f"SHAP {plot_type.capitalize()} Plot")
        results_window.geometry("1200x900")
        results_window.configure(bg=BACKGROUND_COLOR)
        
        # Position the window
        results_window.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

        # Main container with stylish border
        main_frame = tk.Frame(results_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text=f"SHAP {plot_type.capitalize()} Plot", 
                             font=("Helvetica", 18, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
        title_label.pack(pady=(10, 20))
        
        # Create figure with improved styling
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Generate the selected plot
        if plot_type == "summary":
            shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
            ax.set_title("SHAP Feature Importance", fontsize=16, pad=20)
        elif plot_type == "dependence":
            # Use the feature with highest importance for dependence plot
            feature_importance = np.abs(shap_values.values).mean(axis=0)
            top_feature_idx = np.argmax(feature_importance)
            top_feature = X.columns[top_feature_idx]
            shap.dependence_plot(top_feature, shap_values.values, X_test, 
                                interaction_index=None, ax=ax, show=False)
            ax.set_title(f"SHAP Dependence Plot for {top_feature}", fontsize=16, pad=20)
        
        # Style the plot
        ax.title.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        
        # Add border to the plot
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color(HIGHLIGHT_COLOR)
            spine.set_linewidth(2)
        
        # Embed in Tkinter with stylish frame
        plot_frame = tk.Frame(main_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Close button with modern style
        close_btn = tk.Button(main_frame, text="Close", command=results_window.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                            relief=tk.FLAT, padx=30, pady=8, bd=0, highlightthickness=0,
                            activebackground=HOVER_COLOR)
        close_btn.pack(pady=(10, 20))
        add_hover_effect(close_btn)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error during SHAP analysis: {e}")

def setup_missing_values_tab(tab):
    """Setup the missing values tab with improved styling."""
    tab.configure(style='TFrame')
    
    # Info label
    info_label = tk.Label(tab, text="Select how to handle missing values for each column:", 
                         font=("Helvetica", 12), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    info_label.pack(pady=(10, 20))

    # Scrollable frame for columns
    scroll_frame = tk.Frame(tab, bg=PRIMARY_COLOR)
    scroll_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(scroll_frame, bg=PRIMARY_COLOR, highlightthickness=0)
    scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg=PRIMARY_COLOR)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Get columns with missing values
    null_counts = df.isnull().sum()
    cols_with_missing = null_counts[null_counts > 0].index.tolist()
    
    if not cols_with_missing:
        no_missing_label = tk.Label(scrollable_frame, text="No missing values found in the dataset!", 
                                  font=("Helvetica", 12), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
        no_missing_label.pack(pady=20)
        return

    # Create a frame for each column with missing values
    column_frames = []
    for col in cols_with_missing:
        col_frame = tk.Frame(scrollable_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.GROOVE, padx=10, pady=10)
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

    # Apply button with modern style
        # Apply button with modern style
    apply_btn = tk.Button(tab, text="Apply Changes", command=lambda: apply_missing_value_changes(column_frames),
                         bg=SUCCESS_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground=HOVER_COLOR)
    apply_btn.pack(pady=(20, 10))
    add_hover_effect(apply_btn)

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

def move_items(source_list, dest_list, allow_multiple=True):
    """Move selected items from source list to destination list."""
    selected = source_list.curselection()
    if not selected:
        return
    
    if not allow_multiple and dest_list.size() > 0:
        messagebox.showwarning("Warning", "Only one target variable is allowed")
        return
        
    for idx in selected[::-1]:  # Reverse to maintain order when deleting
        item = source_list.get(idx)
        dest_list.insert(tk.END, item)
        source_list.delete(idx)

def remove_items(*lists):
    """Remove selected items from lists and return them to available features."""
    for lst in lists:
        selected = lst.curselection()
        for idx in selected[::-1]:  # Reverse to maintain order when deleting
            item = lst.get(idx)
            available_list = root.nametowidget(lst.master.master.master.children['!frame'].children['!listbox'])
            available_list.insert(tk.END, item)
            lst.delete(idx)

def apply_feature_selection(features_list, target_list):
    """Apply the selected feature and target variables with proper validation."""
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
        
        # Set X and y (for backward compatibility with single target)
        X = df[features]
        if len(targets) == 1:
            y = df[targets[0]]
        else:
            y = df[targets]  # For multiple targets
            
        messagebox.showinfo("Success", "Feature selection applied successfully!")
        update_data_preview()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to apply feature selection: {e}")

def setup_feature_selection_tab(tab):
    """Setup the feature selection tab with improved styling."""
    tab.configure(style='TFrame')
    
    # Info label
    info_label = tk.Label(tab, text="Select features and target variable(s):", 
                         font=("Helvetica", 12), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    info_label.pack(pady=(10, 20))

    # Main frame for feature selection
    selection_frame = tk.Frame(tab, bg=PRIMARY_COLOR)
    selection_frame.pack(fill=tk.BOTH, expand=True)

    # Available features frame with stylish border
    available_frame = tk.Frame(selection_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
    available_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

    available_label = tk.Label(available_frame, text="Available Features", 
                             font=("Helvetica", 11, "bold"), fg=HIGHLIGHT_COLOR, bg=SECONDARY_COLOR)
    available_label.pack(pady=5)

    # Listbox for available features
    available_list = tk.Listbox(available_frame, selectmode=tk.MULTIPLE, 
                              bg=PRIMARY_COLOR, fg=TEXT_COLOR, font=("Helvetica", 10),
                              selectbackground=ACCENT_COLOR, selectforeground=TEXT_COLOR,
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
                              bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Helvetica", 10, "bold"),
                              relief=tk.FLAT, padx=10, pady=5, bd=0, highlightthickness=0,
                              activebackground=HOVER_COLOR)
    add_feature_btn.pack(pady=10)
    add_hover_effect(add_feature_btn)

    # Add to target button
    add_target_btn = tk.Button(button_frame, text="→ Target →", 
                             command=lambda: move_items(available_list, target_list, allow_multiple=False),
                             bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Helvetica", 10, "bold"),
                             relief=tk.FLAT, padx=10, pady=5, bd=0, highlightthickness=0,
                             activebackground=HOVER_COLOR)
    add_target_btn.pack(pady=10)
    add_hover_effect(add_target_btn)

    # Remove button
    remove_btn = tk.Button(button_frame, text="← Remove ←", 
                          command=lambda: remove_items(features_list, target_list),
                          bg=DANGER_COLOR, fg=TEXT_COLOR, font=("Helvetica", 10, "bold"),
                          relief=tk.FLAT, padx=10, pady=5, bd=0, highlightthickness=0,
                          activebackground="#B71C1C")
    remove_btn.pack(pady=10)
    add_hover_effect(remove_btn)

    # Selected features frame with stylish border
    selected_frame = tk.Frame(selection_frame, bg=PRIMARY_COLOR)
    selected_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Features list frame
    features_frame = tk.Frame(selected_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
    features_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    features_label = tk.Label(features_frame, text="Input Features", 
                            font=("Helvetica", 11, "bold"), fg=HIGHLIGHT_COLOR, bg=SECONDARY_COLOR)
    features_label.pack(pady=5)

    features_list = tk.Listbox(features_frame, selectmode=tk.MULTIPLE, 
                             bg=PRIMARY_COLOR, fg=TEXT_COLOR, font=("Helvetica", 10),
                             selectbackground=ACCENT_COLOR, selectforeground=TEXT_COLOR,
                             highlightthickness=0, bd=0)
    features_scroll = ttk.Scrollbar(features_frame, orient="vertical", command=features_list.yview)
    features_list.configure(yscrollcommand=features_scroll.set)
    
    features_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    features_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Target frame with stylish border
    target_frame = tk.Frame(selected_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
    target_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    target_label = tk.Label(target_frame, text="Target Variable", 
                          font=("Helvetica", 11, "bold"), fg=HIGHLIGHT_COLOR, bg=SECONDARY_COLOR)
    target_label.pack(pady=5)

    target_list = tk.Listbox(target_frame, selectmode=tk.SINGLE, 
                           bg=PRIMARY_COLOR, fg=TEXT_COLOR, font=("Helvetica", 10),
                           selectbackground=ACCENT_COLOR, selectforeground=TEXT_COLOR,
                           highlightthickness=0, bd=0)
    target_scroll = ttk.Scrollbar(target_frame, orient="vertical", command=target_list.yview)
    target_list.configure(yscrollcommand=target_scroll.set)
    
    target_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    target_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Apply button with modern style
    apply_btn = tk.Button(tab, text="Apply Feature Selection", 
                        command=lambda: apply_feature_selection(features_list, target_list),
                        bg=SUCCESS_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                        relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                        activebackground=HOVER_COLOR)
    apply_btn.pack(pady=(20, 10))
    add_hover_effect(apply_btn)

def setup_string_conversion_tab(tab):
    """Setup the string to numeric conversion tab."""
    tab.configure(style='TFrame')
    
    # Info label
    info_label = tk.Label(tab, text="Select columns to convert from string to numeric:", 
                         font=("Helvetica", 12), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    info_label.pack(pady=(10, 20))

    # Main frame for conversion
    conversion_frame = tk.Frame(tab, bg=PRIMARY_COLOR)
    conversion_frame.pack(fill=tk.BOTH, expand=True)

    # Available columns frame
    available_frame = tk.Frame(conversion_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
    available_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

    available_label = tk.Label(available_frame, text="String Columns", 
                             font=("Helvetica", 11, "bold"), fg=HIGHLIGHT_COLOR, bg=SECONDARY_COLOR)
    available_label.pack(pady=5)

    # Listbox for string columns
    string_list = tk.Listbox(available_frame, selectmode=tk.MULTIPLE, 
                           bg=PRIMARY_COLOR, fg=TEXT_COLOR, font=("Helvetica", 10),
                           selectbackground=ACCENT_COLOR, selectforeground=TEXT_COLOR,
                           highlightthickness=0, bd=0)
    scrollbar = ttk.Scrollbar(available_frame, orient="vertical", command=string_list.yview)
    string_list.configure(yscrollcommand=scrollbar.set)
    
    # Populate with string columns
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in string_cols:
        string_list.insert(tk.END, col)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    string_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Conversion options frame
    options_frame = tk.Frame(conversion_frame, bg=PRIMARY_COLOR)
    options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

    # Conversion type label
    type_label = tk.Label(options_frame, text="Conversion Type:", 
                         font=("Helvetica", 10, "bold"), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    type_label.pack(pady=(20, 5))

    # Radio buttons for conversion type
    conversion_var = tk.StringVar(value="numeric")

    numeric_radio = tk.Radiobutton(options_frame, text="To Numeric", 
                                 variable=conversion_var, value="numeric", 
                                 bg=PRIMARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                                 activebackground=PRIMARY_COLOR, activeforeground=TEXT_COLOR,
                                 font=("Helvetica", 10))
    numeric_radio.pack(anchor=tk.W)

    category_radio = tk.Radiobutton(options_frame, text="To Category Codes", 
                                  variable=conversion_var, value="category", 
                                  bg=PRIMARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                                  activebackground=PRIMARY_COLOR, activeforeground=TEXT_COLOR,
                                  font=("Helvetica", 10))
    category_radio.pack(anchor=tk.W)

    # Apply button
    apply_btn = tk.Button(tab, text="Convert Selected", 
                        command=lambda: apply_string_conversion(string_list, conversion_var.get()),
                        bg=SUCCESS_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                        relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                        activebackground=HOVER_COLOR)
    apply_btn.pack(pady=(20, 10))
    add_hover_effect(apply_btn)

def apply_string_conversion(string_list, conversion_type):
    """Apply the string to numeric conversion."""
    global df
    
    try:
        selected_cols = [string_list.get(i) for i in string_list.curselection()]
        if not selected_cols:
            messagebox.showwarning("Warning", "Please select at least one column to convert")
            return
            
        for col in selected_cols:
            if conversion_type == "numeric":
                # Try to convert to numeric, coerce errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:  # category
                # Convert to category codes
                df[col] = df[col].astype('category').cat.codes
                
        messagebox.showinfo("Success", f"Successfully converted {len(selected_cols)} columns")
        update_data_preview()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to convert columns: {e}")

def setup_pca_prep_tab(tab):
    """Setup the PCA preparation tab."""
    tab.configure(style='TFrame')
    
    # Info label
    info_label = tk.Label(tab, text="Configure PCA for your data:", 
                         font=("Helvetica", 12), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    info_label.pack(pady=(10, 20))

    # Main frame for PCA options
    pca_frame = tk.Frame(tab, bg=PRIMARY_COLOR)
    pca_frame.pack(fill=tk.BOTH, expand=True, padx=20)

    # Standardize data checkbox
    standardize_var = tk.BooleanVar(value=True)
    standardize_check = tk.Checkbutton(pca_frame, text="Standardize data before PCA", 
                                     variable=standardize_var, 
                                     bg=PRIMARY_COLOR, fg=TEXT_COLOR, selectcolor=ACCENT_COLOR,
                                     activebackground=PRIMARY_COLOR, activeforeground=TEXT_COLOR,
                                     font=("Helvetica", 10))
    standardize_check.pack(anchor=tk.W, pady=5)

    # Number of components frame
    components_frame = tk.Frame(pca_frame, bg=PRIMARY_COLOR)
    components_frame.pack(fill=tk.X, pady=10)

    components_label = tk.Label(components_frame, text="Number of components:", 
                              font=("Helvetica", 10), fg=TEXT_COLOR, bg=PRIMARY_COLOR)
    components_label.pack(side=tk.LEFT)

    components_spin = tk.Spinbox(components_frame, from_=1, to=min(df.shape[0], df.shape[1]), 
                               font=("Helvetica", 10), bg=SECONDARY_COLOR, fg=TEXT_COLOR,
                               insertbackground=TEXT_COLOR, highlightthickness=0)
    components_spin.pack(side=tk.LEFT, padx=10)
    components_spin.delete(0, tk.END)
    components_spin.insert(0, "2")

    # Apply button
    apply_btn = tk.Button(tab, text="Apply PCA", 
                        command=lambda: apply_pca_prep(standardize_var.get(), int(components_spin.get())),
                        bg=SUCCESS_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                        relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                        activebackground=HOVER_COLOR)
    apply_btn.pack(pady=(20, 10))
    add_hover_effect(apply_btn)

def apply_pca_prep(standardize, n_components):
    """Apply PCA transformation to the data."""
    global df
    
    try:
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            messagebox.showerror("Error", "No numeric columns found for PCA")
            return
            
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
        else:
            scaled_data = numeric_df.values
            
        # Apply PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        
        # Create new column names
        pc_columns = [f"PC{i+1}" for i in range(n_components)]
        
        # Create new DataFrame with PCA components
        pca_df = pd.DataFrame(data=principal_components, columns=pc_columns)
        
        # Add non-numeric columns back if any
        non_numeric_cols = df.select_dtypes(exclude=[np.number])
        if not non_numeric_cols.empty:
            pca_df = pd.concat([pca_df, non_numeric_cols.reset_index(drop=True)], axis=1)
            
        # Update the global DataFrame
        df = pca_df
        
        messagebox.showinfo("Success", f"PCA applied successfully! Created {n_components} principal components.")
        update_data_preview()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to apply PCA: {e}")
def show_prepare_data_interface():
    """Show interface for data preparation with improved styling."""
    prep_window = tk.Toplevel(root)
    prep_window.title("Prepare Data")
    prep_window.geometry("1100x800")
    prep_window.configure(bg=BACKGROUND_COLOR)
    
    # Position the window
    prep_window.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

    # Main container with stylish border
    main_frame = tk.Frame(prep_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Title
    title_label = tk.Label(main_frame, text="Data Preparation", 
                         font=("Helvetica", 18, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
    title_label.pack(pady=(10, 20))

    # Tab control with modern style
    style = ttk.Style()
    style.configure('TNotebook', background=PRIMARY_COLOR)
    style.configure('TNotebook.Tab', background=SECONDARY_COLOR, foreground=TEXT_COLOR,
                   font=('Helvetica', 10, 'bold'), padding=[10, 5])
    style.map('TNotebook.Tab', background=[('selected', ACCENT_COLOR)], foreground=[('selected', TEXT_COLOR)])
    
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

    # Tab 3: String to Numeric Conversion
    tab3 = ttk.Frame(tab_control)
    tab_control.add(tab3, text="String Conversion")
    setup_string_conversion_tab(tab3)

    # Tab 4: PCA Preparation
    tab4 = ttk.Frame(tab_control)
    tab_control.add(tab4, text="PCA Preparation")
    setup_pca_prep_tab(tab4)

    # Close button with modern style
    close_btn = tk.Button(main_frame, text="Close", command=prep_window.destroy,
                         bg=DANGER_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=30, pady=8, bd=0, highlightthickness=0,
                         activebackground="#B71C1C")
    close_btn.pack(pady=(10, 20))
    add_hover_effect(close_btn)
    
def plot_graph(graph_type):
    """Function to plot selected graph with improved styling."""
    if df.empty:
        messagebox.showerror("Error", "Dataset is empty. Please load a valid dataset.")
        return

    try:
        # Create a new window for the plot
        plot_window = tk.Toplevel(root)
        plot_window.title(f"{graph_type} Visualization")
        plot_window.geometry("900x700")
        plot_window.configure(bg=BACKGROUND_COLOR)
        
        # Position the window
        plot_window.geometry(f"+{root.winfo_x()+50}+{root.winfo_y()+50}")

        # Main container with stylish border
        main_frame = tk.Frame(plot_window, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(main_frame, text=f"{graph_type} Visualization", 
                             font=("Helvetica", 18, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
        title_label.pack(pady=(10, 20))

        # Create figure with improved styling
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # Set the facecolor for the figure and axes
        fig.patch.set_facecolor(PRIMARY_COLOR)
        ax.set_facecolor(PRIMARY_COLOR)

        if graph_type == "Pie Chart":
            counts = df.iloc[:, -1].value_counts()
            colors = [ACCENT_COLOR, HOVER_COLOR, "#B39DDB", "#D1C4E9"]
            wedges, texts, autotexts = ax.pie(counts, labels=counts.index, autopct='%1.1f%%', shadow = True, 
                  colors=colors[:len(counts)], textprops={'color': TEXT_COLOR})
            
            # Improve pie chart styling
            for w in wedges:
                w.set_edgecolor(PRIMARY_COLOR)
                w.set_linewidth(0.5)
                
            for t in texts + autotexts:
                t.set_fontsize(10)
                
            ax.set_title("Class Distribution", fontsize=16, pad=20, color=TEXT_COLOR)

        elif graph_type == "Bar Chart":
            counts = df.iloc[:, -1].value_counts()
            bars = ax.bar(counts.index, counts.values, color=ACCENT_COLOR, edgecolor=HIGHLIGHT_COLOR)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height}', ha='center', va='bottom', color=TEXT_COLOR)
                
                
            ax.set_title("Class Frequency", fontsize=16, color=TEXT_COLOR)
            ax.set_xlabel("Classes", fontsize=12, color=TEXT_COLOR)
            ax.set_ylabel("Frequency", fontsize=12, color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        elif graph_type == "Scatter Plot":
            if X.shape[1] < 2:
                messagebox.showerror("Error", "Scatter plot requires at least two features.")
                plot_window.destroy()
                return
                
            scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='viridis', 
                               edgecolors='k', alpha=0.7)
            ax.set_title("Feature Relationship", fontsize=16, color=TEXT_COLOR)
            ax.set_xlabel("Feature 1", fontsize=12, color=TEXT_COLOR)
            ax.set_ylabel("Feature 2", fontsize=12, color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            
            # Add colorbar with styling
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("Classes", fontsize=12, color=TEXT_COLOR)
            cbar.ax.tick_params(colors=TEXT_COLOR)
            cbar.outline.set_edgecolor(TEXT_COLOR)

        elif graph_type == "Histogram":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) == 0:
                messagebox.showerror("Error", "No numeric columns found for histogram.")
                plot_window.destroy()
                return
                
            fig.clf()
            ax = fig.subplots(1, 1)
            ax.set_facecolor(PRIMARY_COLOR)
            
            for column in numeric_df.columns:
                sns.histplot(data=numeric_df, x=column, kde=False, 
                            color=ACCENT_COLOR, alpha=0.5, label=column, ax=ax,
                            edgecolor=HIGHLIGHT_COLOR)
            
            ax.set_title("Feature Distributions", fontsize=16, color=TEXT_COLOR)
            ax.set_xlabel("Value", fontsize=12, color=TEXT_COLOR)
            ax.set_ylabel("Frequency", fontsize=12, color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.legend(facecolor=PRIMARY_COLOR, edgecolor=PRIMARY_COLOR, 
                     labelcolor=TEXT_COLOR)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        elif graph_type == "KDE Plot":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) == 0:
                messagebox.showerror("Error", "No numeric columns found for KDE plot.")
                plot_window.destroy()
                return
                
            fig.clf()
            ax = fig.subplots(1, 1)
            ax.set_facecolor(PRIMARY_COLOR)
            
            for column in numeric_df.columns:
                sns.kdeplot(data=numeric_df, x=column, 
                           color=ACCENT_COLOR, alpha=0.7, label=column, ax=ax,
                           linewidth=2)
            
            ax.set_title("Density Estimation", fontsize=16, color=TEXT_COLOR)
            ax.set_xlabel("Value", fontsize=12, color=TEXT_COLOR)
            ax.set_ylabel("Density", fontsize=12, color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.legend(facecolor=PRIMARY_COLOR, edgecolor=PRIMARY_COLOR, 
                     labelcolor=TEXT_COLOR)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        elif graph_type == "Line Plot":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) == 0:
                messagebox.showerror("Error", "No numeric columns found for line plot.")
                plot_window.destroy()
                return
                
            fig.clf()
            ax = fig.subplots(1, 1)
            ax.set_facecolor(PRIMARY_COLOR)
            
            for column in numeric_df.columns:
                sns.lineplot(data=numeric_df, x=numeric_df.index, y=column, 
                           color=ACCENT_COLOR, alpha=0.8, label=column, ax=ax,
                           linewidth=2)
            
            ax.set_title("Line Plot", fontsize=16, color=TEXT_COLOR)
            ax.set_xlabel("Index", fontsize=12, color=TEXT_COLOR)
            ax.set_ylabel("Value", fontsize=12, color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.legend(facecolor=PRIMARY_COLOR, edgecolor=PRIMARY_COLOR, 
                     labelcolor=TEXT_COLOR)
            ax.grid(True, linestyle='--', alpha=0.3)

        elif graph_type == "Area Plot":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) == 0:
                messagebox.showerror("Error", "No numeric columns found for area plot.")
                plot_window.destroy()
                return
                
            fig.clf()
            ax = fig.subplots(1, 1)
            ax.set_facecolor(PRIMARY_COLOR)
            
            numeric_df.plot.area(ax=ax, alpha=0.7, color=[ACCENT_COLOR, HOVER_COLOR, "#B39DDB"])
            ax.set_title("Area Plot", fontsize=16, color=TEXT_COLOR)
            ax.set_xlabel("Index", fontsize=12, color=TEXT_COLOR)
            ax.set_ylabel("Value", fontsize=12, color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.legend(facecolor=PRIMARY_COLOR, edgecolor=PRIMARY_COLOR, 
                     labelcolor=TEXT_COLOR)
            ax.grid(True, linestyle='--', alpha=0.3)

        elif graph_type == "Box Plot":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) == 0:
                messagebox.showerror("Error", "No numeric columns found for box plot.")
                plot_window.destroy()
                return
                
            fig.clf()
            ax = fig.subplots(1, 1)
            ax.set_facecolor(PRIMARY_COLOR)
            
            numeric_df.plot.box(ax=ax, patch_artist=True, 
                              boxprops=dict(facecolor=ACCENT_COLOR, color=TEXT_COLOR),
                              whiskerprops=dict(color=TEXT_COLOR),
                              capprops=dict(color=TEXT_COLOR),
                              medianprops=dict(color='red'),
                              flierprops=dict(marker='o', markersize=5,
                                            markerfacecolor=ACCENT_COLOR,
                                            markeredgecolor=TEXT_COLOR))
            ax.set_title("Box Plot", fontsize=16, color=TEXT_COLOR)
            ax.set_ylabel("Value", fontsize=12, color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.grid(True, linestyle='--', alpha=0.3)

        # Style the plot borders
        for spine in ax.spines.values():
            spine.set_color(HIGHLIGHT_COLOR)
            spine.set_linewidth(2)
        
        # Embed in Tkinter with stylish frame
        plot_frame = tk.Frame(main_frame, bg=SECONDARY_COLOR, bd=1, relief=tk.SUNKEN)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Close button with modern style
        close_btn = tk.Button(main_frame, text="Close", command=plot_window.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                            relief=tk.FLAT, padx=30, pady=8, bd=0, highlightthickness=0,
                            activebackground=HOVER_COLOR)
        close_btn.pack(pady=(10, 20))
        add_hover_effect(close_btn)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to plot graph: {e}")

def show_visualization_options():
    """Show visualization options popup with improved styling."""
    popup = tk.Toplevel(root)
    popup.title("Visualization Options")
    popup.geometry("700x600")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)
    
    # Position the popup
    popup.geometry(f"+{root.winfo_x()+root.winfo_width()//2-350}+{root.winfo_y()+root.winfo_height()//2-300}")

    # Stylish content frame
    content_frame = tk.Frame(popup, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
    content_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

    label = tk.Label(content_frame, text="Select Visualization Type", 
                    font=("Helvetica", 16, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
    label.pack(pady=(20, 20))

    # Create a frame for the buttons
    btn_frame = tk.Frame(content_frame, bg=PRIMARY_COLOR)
    btn_frame.pack(pady=10, fill=tk.BOTH, expand=True)

    # Configure button style
    btn_style = {
        'bg': SECONDARY_COLOR,
        'fg': TEXT_COLOR,
        'font': ("Helvetica", 11, "bold"),
        'relief': tk.FLAT,
        'bd': 0,
        'highlightthickness': 0,
        'activebackground': HOVER_COLOR,
        'padx': 20,
        'pady': 12,
        'width': 18
    }

    # First row of buttons
    row1_frame = tk.Frame(btn_frame, bg=PRIMARY_COLOR)
    row1_frame.pack(pady=5)

    pie_btn = tk.Button(row1_frame, text="Pie Chart", command=lambda: [popup.destroy(), plot_graph("Pie Chart")],
                       **btn_style)
    pie_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(pie_btn)

    bar_btn = tk.Button(row1_frame, text="Bar Chart", command=lambda: [popup.destroy(), plot_graph("Bar Chart")],
                       **btn_style)
    bar_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(bar_btn)

    # Second row of buttons
    row2_frame = tk.Frame(btn_frame, bg=PRIMARY_COLOR)
    row2_frame.pack(pady=5)

    scatter_btn = tk.Button(row2_frame, text="Scatter Plot", command=lambda: [popup.destroy(), plot_graph("Scatter Plot")],
                          **btn_style)
    scatter_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(scatter_btn)

    hist_btn = tk.Button(row2_frame, text="Histogram", command=lambda: [popup.destroy(), plot_graph("Histogram")],
                       **btn_style)
    hist_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(hist_btn)

    # Third row of buttons
    row3_frame = tk.Frame(btn_frame, bg=PRIMARY_COLOR)
    row3_frame.pack(pady=5)

    kde_btn = tk.Button(row3_frame, text="KDE Plot", command=lambda: [popup.destroy(), plot_graph("KDE Plot")],
                      **btn_style)
    kde_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(kde_btn)

    line_btn = tk.Button(row3_frame, text="Line Plot", command=lambda: [popup.destroy(), plot_graph("Line Plot")],
                       **btn_style)
    line_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(line_btn)

    # Fourth row of buttons
    row4_frame = tk.Frame(btn_frame, bg=PRIMARY_COLOR)
    row4_frame.pack(pady=5)

    area_btn = tk.Button(row4_frame, text="Area Plot", command=lambda: [popup.destroy(), plot_graph("Area Plot")],
                       **btn_style)
    area_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(area_btn)

    box_btn = tk.Button(row4_frame, text="Box Plot", command=lambda: [popup.destroy(), plot_graph("Box Plot")],
                      **btn_style)
    box_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(box_btn)

    # Close button with modern style
    close_btn = tk.Button(content_frame, text="Close", command=popup.destroy,
                         bg=DANGER_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=8, bd=0, highlightthickness=0,
                         activebackground="#B71C1C")
    close_btn.pack(pady=(20, 10))
    add_hover_effect(close_btn, hover_color=PRIMARY_COLOR)

# Main application window
root = tk.Tk()
root.title("Machine Learning Visualization Dashboard")
root.geometry("1400x900")
root.configure(bg=BACKGROUND_COLOR)

# Apply modern theme
style = ttk.Style()
style.theme_use('clam')

# Configure styles
style.configure('TFrame', background=BACKGROUND_COLOR)
style.configure('TLabel', background=BACKGROUND_COLOR, foreground=TEXT_COLOR, font=('Helvetica', 10))
style.configure('TButton', background=ACCENT_COLOR, foreground=TEXT_COLOR, 
               font=('Helvetica', 10, 'bold'), borderwidth=0)
style.map('TButton', background=[('active', HOVER_COLOR)])
style.configure('TNotebook', background=BACKGROUND_COLOR)
style.configure('TNotebook.Tab', background=SECONDARY_COLOR, foreground=TEXT_COLOR,
               font=('Helvetica', 10, 'bold'), padding=[10, 5])
style.map('TNotebook.Tab', background=[('selected', ACCENT_COLOR)], foreground=[('selected', TEXT_COLOR)])

# Main container frame
main_container = tk.Frame(root, bg=BACKGROUND_COLOR)
main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Header frame with stylish border
header_frame = tk.Frame(main_container, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
header_frame.pack(fill=tk.X, pady=(0, 20))

# Title label with gradient effect
title_frame = tk.Frame(header_frame, bg=PRIMARY_COLOR)
title_frame.pack(fill=tk.X, pady=10)

main_label = tk.Label(title_frame, 
                     text="INSIGHT_PREDICT: ML DATA ANALYZER",
                     font=("Helvetica", 24, "bold"), 
                     fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
main_label.pack(pady=(0, 5))

# Subtitle
sub_label = tk.Label(title_frame, 
                    text="Explore, Analyze and Visualize Your Data",
                    font=("Helvetica", 12), 
                    fg=TEXT_COLOR, bg=PRIMARY_COLOR)
sub_label.pack()

# Separator
separator = ttk.Separator(header_frame, orient='horizontal', style='TSeparator')
separator.pack(fill=tk.X, pady=10)

# Content frame
content_frame = tk.Frame(main_container, bg=BACKGROUND_COLOR)
content_frame.pack(fill=tk.BOTH, expand=True)

# Left panel (buttons) with stylish border
left_panel = tk.Frame(content_frame, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

# Configure button style
btn_style = {
    'bg': ACCENT_COLOR,
    'fg': TEXT_COLOR,
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
load_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(load_btn)

# Prepare Data button
prepare_data_btn = tk.Button(left_panel, text="Prepare Data", command=show_prepare_data_interface,
                           **btn_style, state=tk.DISABLED)
prepare_data_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(prepare_data_btn)

# Visualize button
visualize_btn = tk.Button(left_panel, text="Visualize Data", command=show_visualization_options,
                        **btn_style, state=tk.DISABLED)
visualize_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(visualize_btn)

# Analyze button
analyze_btn = tk.Button(left_panel, text="Analyze Data", command=show_regression_options,
                      **btn_style, state=tk.DISABLED)
analyze_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(analyze_btn)

# Correlation Matrix button
corr_matrix_btn = tk.Button(left_panel, text="Correlation Matrix", command=show_correlation_matrix,
                          **btn_style, state=tk.DISABLED)
corr_matrix_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(corr_matrix_btn)

# Boosting Algorithms button
boosting_btn = tk.Button(left_panel, text="Boosting Algorithms", command=show_boosting_options,
                       bg=INFO_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                       relief=tk.FLAT, padx=30, pady=15, bd=0, highlightthickness=0,
                       activebackground=HOVER_COLOR, state=tk.DISABLED)
boosting_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(boosting_btn)

# Advanced Analysis button
advanced_btn = tk.Button(left_panel, text="Advanced Analysis", command=show_advanced_analysis_options,
                       bg=INFO_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                       relief=tk.FLAT, padx=30, pady=15, bd=0, highlightthickness=0,
                       activebackground=HOVER_COLOR, state=tk.DISABLED)
advanced_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(advanced_btn)

# Right panel (data preview and info) with stylish border
right_panel = tk.Frame(content_frame, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Data preview label
preview_label = tk.Label(right_panel, text="Dataset Overview", 
                        font=("Helvetica", 14, "bold"), fg=HIGHLIGHT_COLOR, bg=PRIMARY_COLOR)
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
data_preview.tag_config("header", foreground=HIGHLIGHT_COLOR, font=("Helvetica", 12, "bold"))
data_preview.tag_config("info", foreground="#B0BEC5", font=("Helvetica", 10))
data_preview.tag_config("data", foreground=TEXT_COLOR, font=("Courier", 10))
data_preview.tag_config("warning", foreground="#FFA000", font=("Helvetica", 10))

data_preview.insert(tk.END, "No data loaded. Please click 'Load Dataset' to begin.", "info")
data_preview.config(state=tk.DISABLED)

# Footer with stylish border
footer_frame = tk.Frame(main_container, bg=PRIMARY_COLOR, bd=2, relief=tk.GROOVE)
footer_frame.pack(fill=tk.X, pady=(20, 0))

# Exit button with modern style
exit_btn = tk.Button(footer_frame, text="Exit", command=root.quit,
                    bg=DANGER_COLOR, fg=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                    relief=tk.FLAT, padx=30, pady=10, bd=0, highlightthickness=0,
                    activebackground="#B71C1C")
exit_btn.pack(pady=10)
add_hover_effect(exit_btn)

# Initialize empty DataFrame
df = pd.DataFrame()
original_df = pd.DataFrame()
X = pd.DataFrame()
y = pd.Series()

# Start the application
root.mainloop()
