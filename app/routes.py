from flask import Blueprint, render_template, request, redirect, url_for
from .forms import PatientForm
from . import mysql
from src.models.predict_model import predict_single

main = Blueprint('main', __name__)


@main.route('/', methods=['GET', 'POST'])
def index():
    form = PatientForm()
    prediction = None

    if form.validate_on_submit():
        data = {field.name: field.data for field in form}
        data.pop('submit')

        # Convert smoking_status from string to boolean for the model if needed
        # Adjust this based on your model's expected format
        if 'smoking_status' in data:
            if data['smoking_status'] == 'Current Smoker':
                data['smoking_status'] = True
            else:
                data['smoking_status'] = False

        predicted_days = predict_single(data)
        prediction = int(predicted_days)

        # Store to DB - convert boolean back to string for database storage
        smoking_status_for_db = form.smoking_status.data  # Keep original string value
        
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO patients (
                name, age, bmi, blood_pressure, heart_rate, procedures_count, duration_of_treatment,
                gender, admission_reason, admission_type, ward_type, treatment_type, medication_given,
                diagnosis, smoking_status, complications, severity, admission_date, preexisting_condition,
                predicted_recovery_days
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            form.name.data, form.age.data, form.bmi.data, form.blood_pressure.data, form.heart_rate.data,
            form.procedures_count.data, form.duration_of_treatment.data, form.gender.data,
            form.admission_reason.data, form.admission_type.data, form.ward_type.data,
            form.treatment_type.data, form.medication_given.data, form.diagnosis.data,
            smoking_status_for_db, form.complications.data, form.severity.data,
            form.admission_date.data, form.preexisting_condition.data, prediction
        ))
        mysql.connection.commit()
        cur.close()

        return render_template('result.html', name=form.name.data, days=prediction)

    return render_template('index.html', form=form)