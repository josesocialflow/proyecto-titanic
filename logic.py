import joblib
import pandas as pd


def fmt_num(val, decimales=0):
    """Formatea un número con punto para miles y coma para decimales (Formato Hispano)."""
    if pd.isnull(val):
        return "-"
    texto = f"{val:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def preprocess_data(df):
    """Aplica la limpieza, imputación y feature engineering al DataFrame."""
    df = df.copy()

    # Imputación segura de Embarked y creación de Deck
    mode_embarked = df['Embarked'].mode().iloc[0]
    df['Embarked'] = df['Embarked'].fillna(mode_embarked)
    df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')

    # Extraer Title e imputar Age (utilizando raw string r'...')
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    mapa_titulos = {
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
        'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
        'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
    }
    df['Title'] = df['Title'].replace(mapa_titulos)
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

    # Feature Engineering
    df['GroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')
    df['Fare_Per_Person'] = df['Fare'] / df['GroupSize']
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Inflación histórica: 1 GBP (1912) ≈ 160 EUR (Actualidad)
    df['Fare_Per_Person_EUR'] = df['Fare_Per_Person'] * 160.0

    return df


def load_model_artifacts(model_path='model_titanic.pkl', columns_path='model_columns.pkl'):
    """Carga el modelo entrenado y la lista de columnas esperadas."""
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    return model, model_columns


def filter_dataframe(df, pclass_list, sex_list, embarked_list):
    """Filtra el DataFrame según las opciones seleccionadas en la barra lateral."""
    return df[
        (df['Pclass'].isin(pclass_list)) &
        (df['Sex'].isin(sex_list)) &
        (df['Embarked'].isin(embarked_list))
    ]


def predict_survival(model, model_columns, pclass, sex, age, title, family_members, fare_eur):
    """Prepara los datos ingresados por el usuario y realiza la predicción de probabilidad."""
    family_size = family_members + 1
    is_alone = 1 if family_size == 1 else 0
    fare_gbp = fare_eur / 160.0

    input_data = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': float(age),
        'Fare_Per_Person': fare_gbp,
        'FamilySize': family_size,
        'IsAlone': is_alone,
        'Title': title
    }])

    # Codificación One-Hot y alineación exacta de columnas
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Cálculo de probabilidad
    proba = model.predict_proba(input_encoded)[0]
    return proba[1] * 100
