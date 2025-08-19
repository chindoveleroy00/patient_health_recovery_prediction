from flask import Blueprint, render_template, request, redirect, url_for
from .forms import PatientForm
from . import db
from sqlalchemy import text
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

        # Using SQLAlchemy with proper parameter binding
        try:
            db.session.execute(text("""
                INSERT INTO patients (
                    name, age, bmi, blood_pressure, heart_rate, procedures_count, duration_of_treatment,
                    gender, admission_reason, admission_type, ward_type, treatment_type, medication_given,
                    diagnosis, smoking_status, complications, severity, admission_date, preexisting_condition,
                    predicted_recovery_days
                ) VALUES (:name, :age, :bmi, :blood_pressure, :heart_rate, :procedures_count, :duration_of_treatment,
                         :gender, :admission_reason, :admission_type, :ward_type, :treatment_type, :medication_given,
                         :diagnosis, :smoking_status, :complications, :severity, :admission_date, :preexisting_condition,
                         :predicted_recovery_days)
            """), {
                'name': form.name.data,
                'age': form.age.data,
                'bmi': form.bmi.data,
                'blood_pressure': form.blood_pressure.data,
                'heart_rate': form.heart_rate.data,
                'procedures_count': form.procedures_count.data,
                'duration_of_treatment': form.duration_of_treatment.data,
                'gender': form.gender.data,
                'admission_reason': form.admission_reason.data,
                'admission_type': form.admission_type.data,
                'ward_type': form.ward_type.data,
                'treatment_type': form.treatment_type.data,
                'medication_given': form.medication_given.data,
                'diagnosis': form.diagnosis.data,
                'smoking_status': smoking_status_for_db,
                'complications': form.complications.data,
                'severity': form.severity.data,
                'admission_date': form.admission_date.data,
                'preexisting_condition': form.preexisting_condition.data,
                'predicted_recovery_days': prediction
            })
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Database error: {e}")
            # Handle error appropriately - maybe flash a message to user

        return render_template('result.html', name=form.name.data, days=prediction)

    return render_template('index.html', form=form)