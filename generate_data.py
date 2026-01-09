import random
import json
import os

# --- Entity Pools ---
# Same entity lists
all_diseases = [
    # Common
    "diabetes type 2",
    "hypertension",
    "acute bronchitis",
    "chronic heart failure",
    "atrial fibrillation",
    "melanoma",
    "pneumonia",
    "gastroesophageal reflux disease",
    "migraine",
    "rheumatoid arthritis",
    "asthma",
    "tuberculosis",
    "lupus",
    "celiac disease",
    "parkinson's disease",
    "multiple sclerosis",
    "epilepsy",
    "chronic kidney disease",
    "alzheimer's disease",
    "breast cancer",
    "prostate cancer",
    "colon cancer",
    "lung cancer",
    "stroke",
    "coronary artery disease",
    "hepatitis b",
    "hepatitis c",
    "influenza",
    "covid-19",
    "bronchiectasis",
    "cystic fibrosis",
    "diverticulitis",
    "endometriosis",
    "fibromyalgia",
    "gout",
    "hypothyroidism",
    "hyperthyroidism",
    "insomnia",
    "irritable bowel syndrome",
    "meningitis",
    "osteoarthritis",
    "osteoporosis",
    "psoriasis",
    "sarcoidosis",
    "sepsis",
    "shingles",
    "sleep apnea",
    "ulcerative colitis",
    "vertigo",
    "anemia",
    "glaucoma",
    "cataracts",
]

all_medications = [
    # Common
    "metformin",
    "lisinopril",
    "amoxicillin",
    "furosemide",
    "warfarin",
    "pembrolizumab",
    "azithromycin",
    "omeprazole",
    "sumatriptan",
    "methotrexate",
    "albuterol",
    "isoniazid",
    "hydroxychloroquine",
    "gluten-free diet",
    "levodopa",
    "interferon beta-1a",
    "carbamazepine",
    "dialysis",
    "acetaminophen",
    "ibuprofen",
    "aspirin",
    "atorvastatin",
    "simvastatin",
    "amlodipine",
    "metoprolol",
    "losartan",
    "gabapentin",
    "hydrochlorothiazide",
    "sertraline",
    "escitalopram",
    "fluoxetine",
    "trazodone",
    "bupropion",
    "alprazolam",
    "clonazepam",
    "lorazepam",
    "zolpidem",
    "tramadol",
    "fentanyl",
    "morphine",
    "oxycodone",
    "prednisone",
    "dexamethasone",
    "montelukast",
    "fluticasone",
    "insulin glargine",
    "insulin lispro",
    "jardiance",
    "ozempic",
    "eliquis",
    "xarelto",
    "clopidogrel",
    "pantoprazole",
    "famotidine",
    "ciprofloxacin",
    "doxycycline",
    "prednisolone",
    "gabapentin",
]

# --- Varied Training Templates (Simulating 100 unique sentences) ---
train_templates = [
    "The patient presented with severe symptoms of {disease}, necessitating immediate treatment with {medication}.",
    "After a thorough examination, the diagnosis of {disease} was confirmed, and the patient was started on a course of {medication}.",
    "Primary complaints include chronic pain associated with {disease}; {medication} has been prescribed to manage the discomfort.",
    "Despite adherence to {medication}, the patient's {disease} shows signs of progression.",
    "History is significant for {disease}, currently managed with daily {medication}.",
    "The physician noted that {disease} is responding well to the new regimen of {medication}.",
    "Due to adverse effects, we are discontinuing {medication} for the treatment of {disease}.",
    "Screening results indicate a high likelihood of {disease}, warranting a trial of {medication}.",
    "The subject has a long-standing history of {disease} and has been on {medication} for five years.",
    "We observed an acute flare-up of {disease} shortly after the patient missed a dose of {medication}.",
    "Management of {disease} typically involves lifestyle changes alongside {medication}.",
    "He denies any family history of {disease} but is currently taking {medication} for prophylaxis.",
    "The lab markers for {disease} have improved significantly since starting {medication}.",
    "Considering the patient's age and comorbidity of {disease}, {medication} is the preferred therapeutic agent.",
    "Upon discharge, the patient received education on managing {disease} and taking {medication} correctly.",
    "Recurrent episodes of {disease} have been effectively controlled by increasing the dosage of {medication}.",
    "Initial presentation was consistent with {disease}; however, response to {medication} was poor.",
    "Following the diagnosis of {disease}, the patient was referred to a specialist and prescribed {medication}.",
    "There is a concern that {medication} interacts with other drugs taken for {disease}.",
    "We are monitoring renal function closely given the long-term use of {medication} for {disease}.",
    "The clinical picture suggests {disease}.",
    "Long-term outcomes for {disease} are generally favorable.",
    "Please refill the prescription for {medication}.",
    "Patient reported missing doses of {medication}.",
    "Severe {disease} detected in the screening.",
    "{medication} 50mg po daily.",
    "Discussed the risks and benefits of {medication}.",
    "Assessment: {disease}, uncontrolled.",
    "Plan: Continue {medication} and follow up in 3 months.",
    "Subjective: Patient complains of fatigue related to {disease}.",
]

# --- Long/Complex Test Templates (Paragraph-like) ---
test_templates = [
    "The patient, a 45-year-old male with a significant past medical history of {disease}, presented to the emergency department complaining of worsening fatigue and shortness of breath. Upon review of systems, he noted that he has been non-compliant with his home medication, {medication}, for the past two weeks due to side effects.",
    "Discharge Summary: The patient was admitted for an acute exacerbation of {disease}. During the hospital stay, symptoms were initially managed with IV fluids and subsequently transitioned to oral {medication}. The patient is stable for discharge and instructed to follow up with primary care for monitoring of {disease}.",
    "Subjective: Patient reports feeling 'unwell' for 3 days. History of present illness is consistent with a flare of {disease}. She mentions she ran out of {medication} 4 days ago. Objective: Vitals stable. Plan: Restart {medication} immediately and schedule lab work to assess specificity of {disease} markers.",
    "Assessment and Plan: # {disease} - largely controlled but recent stress has caused an uptick in symptoms. We will increase the dosage of {medication} from 10mg to 20mg daily. Patient was advised to monitor for side effects. # Hypertension - stable. # Diabetes - diet controlled.",
    "Consultation Note: Requested evaluation for persistent {disease} resistant to first-line therapies. The patient has previously failed multiple interventions. We recommend initiating a trial of {medication}, a biologic agent shown to be effective in refractory cases of {disease}. Risks discussed.",
    "The 62-year-old female patient with a history of {disease} returns for routine follow-up. She states that the {medication} prescribed at the last visit has significantly reduced her pain levels. However, she is concerned about potential long-term interactions between {medication} and her other supplements.",
    "Emergent care note: Patient found unresponsive. Medical alert bracelet indicates {disease}. EMS administered naloxone without effect. Family arrived and confirmed patient takes {medication} for the condition. Transported to ICU for further management of status epilepticus secondary to {disease}.",
    "Pharmacy note: Clarification needed for the prescription of {medication}. The dosage seems high for a patient with renal impairment secondary to {disease}. Please review and confirm if {medication} dose needs adjustment.",
    "Radiology Report: Chest X-ray shows patchy infiltrates consistent with {disease}. Clinical correlation is recommended. Given the imaging findings and clinical presentation, empiric treatment with {medication} should be considered pending culture results.",
    "Dermatology consult: The patient presents with a diffuse rash. Differential diagnosis includes drug eruption from recent initiation of {medication} versus cutanous manifestation of underlying {disease}. Biopsy performed.",
]


def tokenize_and_tag(text, disease_name=None, medication_name=None):
    # Simple tokenization that splits punctuation better
    clean_text = (
        text.replace(".", " . ")
        .replace(",", " , ")
        .replace(";", " ; ")
        .replace(":", " : ")
        .replace("(", " ( ")
        .replace(")", " ) ")
        .replace("'", " ' ")
    )
    raw_tokens = clean_text.split()

    tags = ["O"] * len(raw_tokens)

    def tag_sequence(phrase, label):
        if not phrase:
            return
        phrase_clean = (
            phrase.replace(".", " . ")
            .replace(",", " , ")
            .replace(";", " ; ")
            .replace(":", " : ")
            .replace("(", " ( ")
            .replace(")", " ) ")
            .replace("'", " ' ")
        )
        p_tokens = phrase_clean.split()

        len_p = len(p_tokens)
        for i in range(len(raw_tokens) - len_p + 1):
            if raw_tokens[i : i + len_p] == p_tokens:
                tags[i] = f"B-{label}"
                for j in range(1, len_p):
                    tags[i + j] = f"I-{label}"

    if disease_name:
        tag_sequence(disease_name, "DISEASE")
    if medication_name:
        tag_sequence(medication_name, "MEDICATION")

    return {"tokens": raw_tokens, "tags": tags}


def generate_dataset(num_samples, templates, diseases_pool, medications_pool):
    data = []
    for _ in range(num_samples):
        template = random.choice(templates)
        disease = random.choice(diseases_pool)
        medication = random.choice(medications_pool)

        try:
            sentence = template.format(disease=disease, medication=medication)
            used_disease = disease if disease in sentence else None
            used_medication = medication if medication in sentence else None

            tagged = tokenize_and_tag(sentence, used_disease, used_medication)
            data.append(tagged)
        except Exception as e:
            print(f"Error formatting: {e}")

    return data


def save_data(data, filename):
    with open(filename, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {len(data)} samples to {filename}")


if __name__ == "__main__":
    random.seed(99)

    random.shuffle(all_diseases)
    random.shuffle(all_medications)

    # Create significant overlap (80% overlap)
    # We want most test entities to be seen, but some to be unseen.

    total_d = len(all_diseases)
    total_m = len(all_medications)

    # Train set gets 90% of entities
    split_train_d = int(0.9 * total_d)
    split_train_m = int(0.9 * total_m)

    train_d = all_diseases[:split_train_d]
    train_m = all_medications[:split_train_m]

    # Test set gets a mix: some from train (seen) and some from the remaining 10% (unseen)
    # Let's say test set is drawn from the LAST 50% of the list
    # Since train has first 90%, the overlap will be the range [50% to 90%]
    # And the unseen will be [90% to 100%]

    split_test_start_d = int(0.5 * total_d)
    split_test_start_m = int(0.5 * total_m)

    test_d = all_diseases[split_test_start_d:]
    test_m = all_medications[split_test_start_m:]

    print(f"Train Diseases: {len(train_d)}")
    print(
        f"Test Diseases: {len(test_d)} (Overlap with train: {len(set(train_d).intersection(set(test_d)))})"
    )

    print("Generating Train Data...")
    # Use BOTH simple and complex templates for training
    # This ensures vocabulary coverage for the context words
    all_train_templates = train_templates + test_templates
    train_data = generate_dataset(5000, all_train_templates, train_d, train_m)

    print("Generating Test Data...")
    test_data = generate_dataset(200, test_templates, test_d, test_m)

    save_data(train_data, "train.jsonl")
    save_data(test_data, "test.jsonl")
