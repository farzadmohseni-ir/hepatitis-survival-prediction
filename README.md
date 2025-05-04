# hepatitis-survival-prediction
Machine learning project for predicting survival in hepatitis patients using classification algorithms and medical features.

# 🧬 Hepatitis Survival Prediction


## 📌 Project Goal
The objective of this machine learning project is to predict the **survival status** (death or survival) of patients diagnosed with **hepatitis** based on various clinical and laboratory features.



## 🗂️ Dataset Description

The dataset `hepatitis.csv` contains medical records of hepatitis patients. The target variable indicates the final survival status of each patient.

| Feature Name | Description                             | Type        |
|--------------|-----------------------------------------|-------------|
| target       | Patient status (1 = deceased, 2 = alive) | Categorical |
| age          | Age of the patient                      | Numeric     |
| gender       | Gender (1 = male, 2 = female)           | Binary      |
| steroid      | Steroid medication usage                | Binary      |
| antivirals   | Antiviral medication usage              | Binary      |
| fatique      | Presence of fatigue                     | Binary      |
| malaise      | Feeling of malaise                      | Binary      |
| anorexia     | Anorexia (loss of appetite)             | Binary      |
| liverBig     | Enlarged liver                          | Binary      |
| liverFirm    | Liver firmness                          | Binary      |
| spleen       | Enlarged spleen                         | Binary      |
| spiders      | Spider angiomas                         | Binary      |
| ascites      | Abdominal fluid accumulation            | Binary      |
| varices      | Esophageal varices                      | Binary      |
| histology    | Histological confirmation               | Binary      |
| bilirubin    | Bilirubin level                         | Numeric     |
| alk          | Alkaline phosphatase level              | Numeric     |
| sgot         | SGOT enzyme level                       | Numeric     |
| albu         | Albumin level                           | Numeric     |
| protime      | Prothrombin time                        | Numeric     |



## 🧪 Project Workflow

### A) Data Preprocessing
- Handling missing values
- Converting binary features to appropriate types
- Normalizing numeric features
- Splitting dataset into features (X) and labels (y)

### B) Model Training & Validation

#### 1. Data Splitting:
- 70% for training and validation
- 30% for final testing

#### 2. Cross-Validation:
- **Stratified 5-Fold Cross Validation** to maintain class distribution during training

#### 3. Classification Algorithms:
- 🎲 Random Forest Classifier
- 💡 Support Vector Machine (SVM)
- 🔬 Multi-Layer Perceptron (MLP Neural Network)

#### 4. Evaluation Metrics:
- Accuracy
- Recall
- Precision
- F1 Score
- AUC (Area Under the ROC Curve)
- Confusion Matrix


---

<div dir="rtl">



## 📌 هدف پروژه
در این پروژه، با استفاده از الگوریتم‌های یادگیری ماشین، سعی می‌کنیم وضعیت نهایی بیماران مبتلا به بیماری **هپاتیت** (مرگ یا بقا) را بر اساس ویژگی‌های پزشکی و آزمایشگاهی موجود پیش‌بینی کنیم.



## 🗂️ مشخصات مجموعه داده

مجموعه داده‌ی `hepatitis.csv` شامل اطلاعات بالینی و آزمایشگاهی بیماران مبتلا به هپاتیت می‌باشد. متغیر هدف `target` نشان‌دهنده وضعیت بقا یا فوت بیمار است.

| نام ویژگی     | توضیح ویژگی                                 | نوع متغیر  |
|---------------|----------------------------------------------|-------------|
| target        | وضعیت بیمار (۱ = فوت شده، ۲ = زنده)         | هدف (کلاسه) |
| age           | سن بیمار                                     | عددی        |
| gender        | جنسیت (۱ = مرد، ۲ = زن)                      | باینری      |
| steroid       | مصرف استروئید                                | باینری      |
| antivirals    | مصرف داروهای ضدویروسی                        | باینری      |
| fatique       | وجود خستگی                                   | باینری      |
| malaise       | احساس کسالت                                 | باینری      |
| anorexia      | بی‌اشتهایی                                   | باینری      |
| liverBig      | بزرگ شدن کبد                                 | باینری      |
| liverFirm     | سفتی کبد                                     | باینری      |
| spleen        | بزرگ شدن طحال                                | باینری      |
| spiders       | رگ‌های عنکبوتی                               | باینری      |
| ascites       | آب آوردن شکم                                 | باینری      |
| varices       | واریس مری                                    | باینری      |
| histology     | تأیید از نظر بافت‌شناسی                     | باینری      |
| bilirubin     | سطح بیلی‌روبین                               | عددی        |
| alk           | سطح آلکالن فسفاتاز                          | عددی        |
| sgot          | سطح SGOT                                     | عددی        |
| albu          | سطح آلبومین                                  | عددی        |
| protime       | زمان پروترومبین                             | عددی        |



## 🧪 مراحل انجام پروژه

### الف) پیش‌پردازش داده‌ها
- بررسی و مدیریت داده‌های گمشده
- تبدیل ویژگی‌های باینری به نوع صحیح
- نرمال‌سازی ویژگی‌های عددی
- جداسازی ویژگی‌ها (X) و برچسب‌ها (y)

### ب) آموزش مدل و اعتبارسنجی

#### 1. تقسیم داده‌ها:
- ۷۰٪ داده‌ها برای آموزش و اعتبارسنجی
- ۳۰٪ داده‌ها برای تست نهایی مدل

#### 2. اعتبارسنجی متقابل:
- استفاده از **Stratified 5-Fold Cross Validation** برای حفظ نسبت کلاس‌ها در آموزش و اعتبارسنجی

#### 3. الگوریتم‌های مورد استفاده:
- 🎲 Random Forest Classifier
- 💡 Support Vector Machine (SVM)
- 🔬 Multi-Layer Perceptron (MLP Neural Network)

#### 4. معیارهای ارزیابی:
- Accuracy
- Recall
- Precision
- F1 Score
- AUC (مساحت زیر منحنی ROC)
- Confusion Matrix

