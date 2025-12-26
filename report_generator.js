const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle, 
        WidthType, ShadingType, PageNumber, PageBreak } = require('docx');
const fs = require('fs');

const targetDistImg = fs.readFileSync('/home/claude/mlops-heart-disease/screenshots/target_distribution.png');
const corrHeatmapImg = fs.readFileSync('/home/claude/mlops-heart-disease/screenshots/correlation_heatmap.png');
const featureHistImg = fs.readFileSync('/home/claude/mlops-heart-disease/screenshots/feature_histograms.png');
const rocCurvesImg = fs.readFileSync('/home/claude/mlops-heart-disease/screenshots/roc_curves.png');
const cmLogRegImg = fs.readFileSync('/home/claude/mlops-heart-disease/screenshots/confusion_matrix_logistic_regression.png');
const featureImpImg = fs.readFileSync('/home/claude/mlops-heart-disease/screenshots/feature_importance_random_forest.png');

const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

const doc = new Document({
    styles: {
        default: { document: { run: { font: "Arial", size: 24 } } },
        paragraphStyles: [
            { id: "Title", name: "Title", basedOn: "Normal", run: { size: 56, bold: true, color: "1a5276", font: "Arial" }, paragraph: { spacing: { before: 240, after: 120 }, alignment: AlignmentType.CENTER } },
            { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 32, bold: true, color: "1a5276", font: "Arial" }, paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
            { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, color: "2874a6", font: "Arial" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
        ]
    },
    numbering: { config: [
        { reference: "bullet-list", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
        { reference: "num-list", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]},
    sections: [{
        properties: { page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
        headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "MLOps Assignment Report", size: 20, color: "666666" })] })] }) },
        footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Page ", size: 20 }), new TextRun({ children: [PageNumber.CURRENT], size: 20 }), new TextRun({ text: " of ", size: 20 }), new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 20 })] })] }) },
        children: [
            // Title Page
            new Paragraph({ spacing: { before: 2000 } }),
            new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("MLOps Experimental Learning Assignment")] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 }, children: [new TextRun({ text: "End-to-End ML Model Development, CI/CD, and Production Deployment", size: 28, color: "666666" })] }),
            new Paragraph({ spacing: { before: 800 } }),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Heart Disease Prediction System", size: 36, bold: true })] }),
            new Paragraph({ spacing: { before: 600 } }),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Course: MLOps (S1-25_AIMLCZG523)", size: 24 })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100 }, children: [new TextRun({ text: "BITS Pilani", size: 24 })] }),
            new Paragraph({ spacing: { before: 800 } }),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Total Marks: 50", size: 24, bold: true })] }),
            
            // TOC
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Table of Contents")] }),
            new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("Introduction and Objective")] }),
            new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("Data Acquisition & Exploratory Data Analysis")] }),
            new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("Feature Engineering & Model Development")] }),
            new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("Experiment Tracking with MLflow")] }),
            new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("Model Packaging & Reproducibility")] }),
            new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("CI/CD Pipeline & Automated Testing")] }),
            new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("Model Containerization")] }),
            new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("Production Deployment")] }),
            new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("Monitoring & Logging")] }),
            new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("Conclusion")] }),

            // Section 1
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("1. Introduction and Objective")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("This project implements a complete MLOps pipeline for heart disease prediction using the UCI Heart Disease dataset. The objective is to design, develop, and deploy a scalable and reproducible machine learning solution utilizing modern MLOps best practices.")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("1.1 Problem Statement")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("Build a machine learning classifier to predict the risk of heart disease based on patient health data, and deploy the solution as a cloud-ready, monitored API.")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("1.2 Dataset Overview")] }),
            new Paragraph({ spacing: { after: 100 }, children: [new TextRun("The Heart Disease UCI Dataset contains 14 features including:")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Age, Sex, Chest Pain Type (cp)")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Resting Blood Pressure (trestbps), Serum Cholesterol (chol)")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Fasting Blood Sugar (fbs), Resting ECG Results (restecg)")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Maximum Heart Rate (thalach), Exercise Induced Angina (exang)")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("ST Depression (oldpeak), Slope, Number of Vessels (ca), Thalassemia (thal)")] }),
            new Paragraph({ spacing: { after: 200 }, numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Target: Binary (0 = No Disease, 1 = Disease Present)")] }),

            // Section 2
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("2. Data Acquisition & Exploratory Data Analysis")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.1 Data Download & Preprocessing")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("The dataset was obtained from the UCI Machine Learning Repository. A download script (download_data.py) automates data acquisition. Preprocessing included handling missing values, converting target to binary, and ensuring numeric features.")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.2 Dataset Statistics")] }),
            new Table({ columnWidths: [3000, 6000], rows: [
                new TableRow({ tableHeader: true, children: [
                    new TableCell({ borders: cellBorders, shading: { fill: "1a5276", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Metric", bold: true, color: "FFFFFF" })] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "1a5276", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Value", bold: true, color: "FFFFFF" })] })] })
                ]}),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Total Samples")] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("297")] })] })] }),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Number of Features")] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("13")] })] })] }),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("No Disease (Class 0)")] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("160 (53.9%)")] })] })] }),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Disease Present (Class 1)")] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("137 (46.1%)")] })] })] }),
            ]}),
            new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 400 }, children: [new TextRun("2.3 Target Distribution")] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 200 }, children: [new ImageRun({ type: "png", data: targetDistImg, transformation: { width: 500, height: 210 }, altText: { title: "Target", description: "Target dist", name: "t" } })] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.4 Correlation Heatmap")] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 200 }, children: [new ImageRun({ type: "png", data: corrHeatmapImg, transformation: { width: 420, height: 370 }, altText: { title: "Corr", description: "Correlation", name: "c" } })] }),

            // Section 2 continued
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.5 Feature Distributions")] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 200 }, children: [new ImageRun({ type: "png", data: featureHistImg, transformation: { width: 520, height: 350 }, altText: { title: "Hist", description: "Histograms", name: "h" } })] }),

            // Section 3
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("3. Feature Engineering & Model Development")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.1 Feature Engineering")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("Feature engineering involved StandardScaler for numerical feature normalization, train-test split (80-20) with stratification, and retention of all 13 features for model training.")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.2 Models Trained")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("Three classification models were trained with hyperparameter tuning using GridSearchCV with 5-fold cross-validation:")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Logistic Regression: ", bold: true }), new TextRun("C values [0.01, 0.1, 1, 10], L2 penalty")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Random Forest: ", bold: true }), new TextRun("n_estimators [50, 100, 200], max_depth [5, 10, 15, None]")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Gradient Boosting: ", bold: true }), new TextRun("n_estimators [50, 100], learning_rate [0.01, 0.1, 0.2]")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.3 Model Performance Comparison")] }),
            new Table({ columnWidths: [2200, 1400, 1400, 1400, 1400, 1400], rows: [
                new TableRow({ tableHeader: true, children: [
                    new TableCell({ borders: cellBorders, shading: { fill: "1a5276", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Model", bold: true, color: "FFFFFF", size: 20 })] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "1a5276", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Accuracy", bold: true, color: "FFFFFF", size: 20 })] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "1a5276", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Precision", bold: true, color: "FFFFFF", size: 20 })] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "1a5276", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Recall", bold: true, color: "FFFFFF", size: 20 })] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "1a5276", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "F1", bold: true, color: "FFFFFF", size: 20 })] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "1a5276", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "ROC-AUC", bold: true, color: "FFFFFF", size: 20 })] })] })
                ]}),
                new TableRow({ children: [
                    new TableCell({ borders: cellBorders, shading: { fill: "d5f5e3", type: ShadingType.CLEAR }, children: [new Paragraph({ children: [new TextRun({ text: "Logistic Reg.", bold: true })] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "d5f5e3", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("86.67%")] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "d5f5e3", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("88.46%")] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "d5f5e3", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("82.14%")] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "d5f5e3", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("85.19%")] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "d5f5e3", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "95.09%", bold: true })] })] })
                ]}),
                new TableRow({ children: [
                    new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Random Forest")] })] }),
                    new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("83.33%")] })] }),
                    new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("87.50%")] })] }),
                    new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("75.00%")] })] }),
                    new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("80.77%")] })] }),
                    new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("94.64%")] })] })
                ]}),
                new TableRow({ children: [
                    new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Gradient Boost")] })] }),
                    new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("81.67%")] })] }),
                    new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("84.00%")] })] }),
                    new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("75.00%")] })] }),
                    new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("79.25%")] })] }),
                    new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("90.96%")] })] })
                ]})
            ]}),
            new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 400 }, children: [new TextRun("3.4 ROC Curves Comparison")] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 200 }, children: [new ImageRun({ type: "png", data: rocCurvesImg, transformation: { width: 400, height: 320 }, altText: { title: "ROC", description: "ROC curves", name: "r" } })] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.5 Confusion Matrix - Best Model")] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 200 }, children: [new ImageRun({ type: "png", data: cmLogRegImg, transformation: { width: 350, height: 260 }, altText: { title: "CM", description: "Confusion matrix", name: "cm" } })] }),

            // Section 4
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4. Experiment Tracking with MLflow")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("MLflow was integrated for comprehensive experiment tracking. All model training runs were logged with:")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Parameters: ", bold: true }), new TextRun("Hyperparameters for each model configuration")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Metrics: ", bold: true }), new TextRun("Accuracy, precision, recall, F1-score, ROC-AUC, CV scores")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Artifacts: ", bold: true }), new TextRun("Confusion matrices, ROC curves, feature importance plots")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Models: ", bold: true }), new TextRun("Serialized sklearn models for each run")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.1 MLflow Experiment Structure")] }),
            new Paragraph({ spacing: { after: 100 }, children: [new TextRun("Experiment Name: heart_disease_prediction")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("Tracking URI: file:///mlruns")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.2 Best Model Selection")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("Based on ROC-AUC score (95.09%), Logistic Regression was selected as the best model for deployment. The model achieves excellent discrimination between disease and non-disease cases while maintaining good interpretability.")] }),

            // Section 5
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("5. Model Packaging & Reproducibility")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.1 Saved Artifacts")] }),
            new Paragraph({ spacing: { after: 100 }, children: [new TextRun("The following artifacts were saved for reproducibility:")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "pipeline.pkl: ", bold: true }), new TextRun("Complete sklearn pipeline with scaler and model")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "best_model.pkl: ", bold: true }), new TextRun("Trained Logistic Regression model")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "scaler.pkl: ", bold: true }), new TextRun("Fitted StandardScaler transformer")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "feature_names.pkl: ", bold: true }), new TextRun("List of 13 feature names")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.2 Requirements File")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("A comprehensive requirements.txt file specifies all dependencies with version constraints to ensure reproducibility across environments.")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.3 Preprocessing Pipeline")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("The sklearn Pipeline object encapsulates both preprocessing (StandardScaler) and prediction (LogisticRegression) steps, ensuring consistent transformations.")] }),

            // Section 6
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("6. CI/CD Pipeline & Automated Testing")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("6.1 GitHub Actions Workflow")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("A comprehensive CI/CD pipeline was implemented using GitHub Actions with the following stages:")] }),
            new Table({ columnWidths: [2000, 7000], rows: [
                new TableRow({ tableHeader: true, children: [
                    new TableCell({ borders: cellBorders, shading: { fill: "1a5276", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Stage", bold: true, color: "FFFFFF" })] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "1a5276", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Description", bold: true, color: "FFFFFF" })] })] })
                ]}),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun({ text: "Lint", bold: true })] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Code quality checks with flake8 and black")] })] })] }),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun({ text: "Test", bold: true })] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Unit tests for data, model, and API (pytest)")] })] })] }),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun({ text: "Train", bold: true })] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("EDA and model training with artifact upload")] })] })] }),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun({ text: "Docker", bold: true })] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Build, test, and save Docker image")] })] })] }),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun({ text: "Security", bold: true })] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Vulnerability scanning with Trivy")] })] })] }),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun({ text: "Deploy", bold: true })] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Production deployment (manual trigger)")] })] })] })
            ]}),
            new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 400 }, children: [new TextRun("6.2 Test Coverage")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("45 unit tests covering:")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Data Tests (15): ", bold: true }), new TextRun("File existence, loading, quality, preprocessing")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Model Tests (15): ", bold: true }), new TextRun("File existence, loading, prediction, performance")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "API Tests (15): ", bold: true }), new TextRun("Endpoints, validation, response format")] }),

            // Section 7
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("7. Model Containerization")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("7.1 Docker Configuration")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("The model is containerized using Docker with the following specifications:")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Base Image: ", bold: true }), new TextRun("python:3.11-slim")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Framework: ", bold: true }), new TextRun("FastAPI with uvicorn server")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Port: ", bold: true }), new TextRun("8000")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Health Check: ", bold: true }), new TextRun("Built-in health endpoint monitoring")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Security: ", bold: true }), new TextRun("Non-root user execution")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("7.2 API Endpoints")] }),
            new Table({ columnWidths: [2000, 1500, 5500], rows: [
                new TableRow({ tableHeader: true, children: [
                    new TableCell({ borders: cellBorders, shading: { fill: "1a5276", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Endpoint", bold: true, color: "FFFFFF" })] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "1a5276", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Method", bold: true, color: "FFFFFF" })] })] }),
                    new TableCell({ borders: cellBorders, shading: { fill: "1a5276", type: ShadingType.CLEAR }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Description", bold: true, color: "FFFFFF" })] })] })
                ]}),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("/predict")] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("POST")] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Make heart disease prediction")] })] })] }),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("/health")] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("GET")] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Health check endpoint")] })] })] }),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("/metrics")] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("GET")] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Prometheus metrics")] })] })] }),
                new TableRow({ children: [new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("/docs")] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("GET")] })] }), new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Swagger API documentation")] })] })] })
            ]}),

            // Section 8
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("8. Production Deployment")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("8.1 Kubernetes Configuration")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("The application is deployed to Kubernetes with the following configuration:")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Deployment: ", bold: true }), new TextRun("3 replicas with rolling update strategy")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Service: ", bold: true }), new TextRun("LoadBalancer type for external access")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "HPA: ", bold: true }), new TextRun("Auto-scaling from 2-10 pods based on CPU/memory")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Ingress: ", bold: true }), new TextRun("NGINX ingress controller for routing")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Resources: ", bold: true }), new TextRun("CPU: 100m-500m, Memory: 256Mi-512Mi")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("8.2 Health Probes")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Liveness Probe: ", bold: true }), new TextRun("Checks /health endpoint every 30s")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Readiness Probe: ", bold: true }), new TextRun("Checks /health endpoint every 10s")] }),

            // Section 9
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("9. Monitoring & Logging")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("9.1 Prometheus Metrics")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("The API exposes Prometheus metrics at /metrics endpoint:")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "predictions_total: ", bold: true }), new TextRun("Counter of total predictions by result type")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "prediction_latency_seconds: ", bold: true }), new TextRun("Histogram of prediction latency")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "api_requests_total: ", bold: true }), new TextRun("Counter of requests by endpoint and method")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("9.2 Grafana Dashboard")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("Grafana is configured to visualize request rate, latency percentiles, prediction distribution, error rates, and resource utilization.")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("9.3 Application Logging")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("Structured logging includes request/response logging, prediction results with confidence scores, error tracking, and model health status events.")] }),

            // Section 10
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("10. Conclusion")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("This MLOps project successfully demonstrates end-to-end machine learning pipeline development following industry best practices.")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("10.1 Summary of Achievements")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Model Performance: ", bold: true }), new TextRun("95.09% ROC-AUC with Logistic Regression")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Test Coverage: ", bold: true }), new TextRun("45 unit tests covering data, model, and API")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "CI/CD Pipeline: ", bold: true }), new TextRun("Fully automated with GitHub Actions")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Containerization: ", bold: true }), new TextRun("Production-ready Docker image")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Deployment: ", bold: true }), new TextRun("Kubernetes manifests with auto-scaling")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Monitoring: ", bold: true }), new TextRun("Prometheus metrics with Grafana dashboards")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("10.2 Repository Structure")] }),
            new Paragraph({ spacing: { after: 200 }, children: [new TextRun("All deliverables are organized in a GitHub repository containing source code, Dockerfile, requirements, tests, Kubernetes manifests, and documentation.")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("10.3 Future Improvements")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Implement model versioning and A/B testing")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Add data drift detection and model retraining triggers")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Enhance monitoring with custom business metrics")] }),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Implement feature store for feature management")] }),
            new Paragraph({ spacing: { before: 400 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "--- End of Report ---", italics: true, color: "666666" })] })
        ]
    }]
});

Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync('/home/claude/mlops-heart-disease/MLOps_Assignment_Report.docx', buffer);
    console.log('Report generated: MLOps_Assignment_Report.docx');
});
