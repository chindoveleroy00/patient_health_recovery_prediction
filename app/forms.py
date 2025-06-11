from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, IntegerField, BooleanField, SelectField, DateField, SubmitField
from wtforms.validators import DataRequired

class PatientForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    age = IntegerField('Age', validators=[DataRequired()])
    bmi = FloatField('BMI', validators=[DataRequired()])
    blood_pressure = IntegerField('Blood Pressure', validators=[DataRequired()])
    heart_rate = IntegerField('Heart Rate', validators=[DataRequired()])
    procedures_count = IntegerField('Procedures Count', validators=[DataRequired()])
    duration_of_treatment = IntegerField('Duration of Treatment', validators=[DataRequired()])
    
    # Updated dropdown fields with predefined choices
    gender = SelectField('Gender', choices=[
        ('', 'Select Gender'),
        ('Male', 'Male'), 
        ('Female', 'Female'), 
        ('Other', 'Other')
    ], validators=[DataRequired()])
    
    admission_reason = SelectField('Admission Reason', choices=[
        ('', 'Select Admission Reason'),
        ('Emergency', 'Emergency'),
        ('Routine Surgery', 'Routine Surgery'),
        ('Accident', 'Accident'),
        ('Chronic Condition', 'Chronic Condition'),
        ('Diagnostic Procedure', 'Diagnostic Procedure'),
        ('Follow-up', 'Follow-up'),
        ('Other', 'Other')
    ], validators=[DataRequired()])
    
    admission_type = SelectField('Admission Type', choices=[
        ('', 'Select Admission Type'),
        ('Emergency', 'Emergency'),
        ('Elective', 'Elective'),
        ('Urgent', 'Urgent'),
        ('Day Case', 'Day Case')
    ], validators=[DataRequired()])
    
    ward_type = SelectField('Ward Type', choices=[
        ('', 'Select Ward Type'),
        ('General Ward', 'General Ward'),
        ('ICU', 'ICU'),
        ('CCU', 'CCU'),
        ('Surgical Ward', 'Surgical Ward'),
        ('Medical Ward', 'Medical Ward'),
        ('Pediatric Ward', 'Pediatric Ward'),
        ('Maternity Ward', 'Maternity Ward')
    ], validators=[DataRequired()])
    
    treatment_type = SelectField('Treatment Type', choices=[
        ('', 'Select Treatment Type'),
        ('Surgical', 'Surgical'),
        ('Medical', 'Medical'),
        ('Conservative', 'Conservative'),
        ('Palliative', 'Palliative'),
        ('Rehabilitation', 'Rehabilitation'),
        ('Diagnostic', 'Diagnostic'),
        ('Preventive', 'Preventive')
    ], validators=[DataRequired()])
    
    medication_given = SelectField('Medication Given', choices=[
        ('', 'Select Medication Type'),
        ('Antibiotics', 'Antibiotics'),
        ('Pain Relievers', 'Pain Relievers'),
        ('Anti-inflammatory', 'Anti-inflammatory'),
        ('Cardiovascular', 'Cardiovascular'),
        ('Respiratory', 'Respiratory'),
        ('Neurological', 'Neurological'),
        ('None', 'None')
    ], validators=[DataRequired()])
    
    diagnosis = SelectField('Diagnosis', choices=[
        ('', 'Select Diagnosis'),
        ('Cardiovascular Disease', 'Cardiovascular Disease'),
        ('Respiratory Disease', 'Respiratory Disease'),
        ('Diabetes', 'Diabetes'),
        ('Cancer', 'Cancer'),
        ('Neurological Disorder', 'Neurological Disorder'),
        ('Infection', 'Infection'),
        ('Injury', 'Injury'),
        ('Other', 'Other')
    ], validators=[DataRequired()])
    
    smoking_status = SelectField('Smoking Status', choices=[
        ('', 'Select Smoking Status'),
        ('Non-smoker', 'Non-smoker'),
        ('Former Smoker', 'Former Smoker'),
        ('Current Smoker', 'Current Smoker')
    ], validators=[DataRequired()])
    
    complications = BooleanField('Complications')
    
    severity = SelectField('Severity', choices=[
        ('', 'Select Severity'),
        ('Mild', 'Mild'), 
        ('Moderate', 'Moderate'), 
        ('Severe', 'Severe')
    ], validators=[DataRequired()])
    
    admission_date = DateField('Admission Date', validators=[DataRequired()])
    preexisting_condition = StringField('Preexisting Condition', validators=[DataRequired()])
    submit = SubmitField('Predict Recovery')