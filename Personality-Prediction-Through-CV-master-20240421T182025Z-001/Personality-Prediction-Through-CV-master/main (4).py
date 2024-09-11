import os
import re
from flask import *
from email.message import EmailMessage
import pandas as pd
import smtplib
import numpy as np
from tkinter import *
from tkinter import filedialog
import tkinter.font as font
from tkinter import messagebox
from functools import partial
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import nltk
import docx2txt
from datetime import datetime

nltk.download('stopwords')

file_path = r"D:\pythonProject2\quiz-webapp-flask-main-20240421T172231Z-001\quiz-webapp-flask-main"

def read_user_data(file_path):
    """Read user data from a file and return it as a dictionary."""
    user_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            fields = line.strip().split(',')
            email = fields[1]
            user_data[email] = fields
    return user_data

def append_user_info(file_path, email,  personality, job_profile):
    """Append user information to the file."""
    # Read existing user data
    user_data = read_user_data(file_path)

    # Convert the email to lowercase for case-insensitive comparison
    email = email.lower()

    # Check if the email exists in the user data
    if email in user_data:
        # Retrieve the user's existing information
        existing_info = user_data[email]

        # Update the user's information with the new skills, personality, and job profile
        existing_info.extend([ ",".join(personality), ", ".join(job_profile)])

        # Write the updated user information back to the file
        with open(file_path, 'w') as file:
            for info in user_data.values():
                file.write(",".join(info) + "\n")
    else:
        print("Email not found in user data.")

def write_results_to_file(candidate_name, age, skills, job_profile, phone_no, email_match, personality):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{candidate_name}_results_{current_time}.txt"

    # Specify the full path to the desired directory
    directory = r"D:\pythonProject2\quiz-webapp-flask-main-20240421T172231Z-001\quiz-webapp-flask-main"

    # Combine the directory path with the filename
    filepath = os.path.join(directory, filename)

    # Ensure the directory exists; if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Combine the directory path with the filename
    filepath = os.path.join(directory, filename)

    with open(filepath, "w") as file:
        file.write(f"Candidate Name: {candidate_name}\n")
        file.write(f"Age: {age}\n")
        file.write("Applicant Skills:\n")
        for skill in skills:
            file.write(f"- {skill}\n")
        file.write(f"Job Profile: {job_profile}\n")
        file.write("Candidate Contact:\n")
        for contact in phone_no:
            file.write(f"- {contact}\n")
        file.write("Candidate Email:\n")
        for email in email_match:
            file.write(f"- {email}\n")
        file.write(f"Predicted Personality: {personality}\n")


class train_model:

    def train(self):
        data = pd.read_csv('training_dataset.csv')
        array = data.values

        for i in range(len(array)):
            if array[i][0] == "Male":
                array[i][0] = 1
            else:
                array[i][0] = 0

        df = pd.DataFrame(array)

        maindf = df[[0, 1, 2, 3, 4, 5, 6]]
        mainarray = maindf.values

        temp = df[7]
        train_y = temp.values

        # Initialize and train Random Forest classifier
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_classifier.fit(mainarray, train_y)
        # Calculate accuracy
        train_pred = self.rf_classifier.predict(mainarray)
        accuracy = accuracy_score(train_y, train_pred)
        print("Accuracy:", accuracy)

    def test(self, test_data):
        try:
            test_predict = list()
            for i in test_data:
                test_predict.append(int(i))
            y_pred = self.rf_classifier.predict([test_predict])
            return y_pred
        except:
            print("All Factors For Finding Personality Not Entered!")


def check_type(data):
    if type(data) == str or type(data) == str:
        return str(data).title()
    if type(data) == list or type(data) == tuple:
        str_list = ""
        for i, item in enumerate(data):
            str_list += item + ", "
        return str_list
    else:
        return str(data)


def prediction_result(top, aplcnt_name, cv_path, personality_values):
    "after applying a job"
    top.withdraw()
    applicant_data = {"Candidate Name": aplcnt_name.get(), "CV Location": cv_path}
    candidateName = aplcnt_name.get()
    age = personality_values[1]
    # skills = extract_skills(cv_path)
    phone_no = personality_values[-2]
    email_match = personality_values[-1]
    # personality = predict_personality(personality_values[2:])
    # write_results_to_file(candidateName, age, skills, job_profile, phone_no, email_match, personality)

    print("\n############# Candidate Entered Data #############\n")
    print(applicant_data, personality_values)

    personality = model.test(personality_values)
    print("\n############# Predicted Personality #############\n")

    print("Candidate_personality:", personality)

    print("Candidate_Name:", candidateName)

    print("Candidate_Age:", age)

    # CV Extract process  Start

    # you may read the database from a csv file or some other database
    SKILLS_DB = {
        'Python',
        'Java',
        'JavaScript',
        'C++',
        'C#',
        'Ruby',
        'Swift',
        'PHP',
        'HTML',
        'CSS',
        'SQL',
        'R',
        'MATLAB',
        'Perl',
        'Go',
        'Kotlin',
        'TypeScript',
        'Scala',
        'Shell Scripting',
        'React',
        'Angular',
        'Vue.js',
        'Node.js',
        'Express.js',
        'Django',
        'Flask',
        'Spring',
        'Hibernate',
        'ASP.NET',
        'Bootstrap',
        'jQuery',
        'TensorFlow',
        'PyTorch',
        'Keras',
        'Scikit-learn',
        'Pandas',
        'NumPy',
        'Matplotlib',
        'Seaborn',
        'NLTK',
        'OpenCV',
        'MongoDB',
        'MySQL',
        'PostgreSQL',
        'SQLite',
        'AWS',
        'Azure',
        'Google Cloud Platform',
        'Docker',
        'Kubernetes',
        'Git',
        'Jenkins',
        'Travis CI',
        'JIRA',
        'Confluence',
        'Agile Methodologies',
        'Scrum',
        'Test-Driven Development (TDD)',
        'Continuous Integration/Continuous Deployment (CI/CD)',
        'Machine Learning',
        'Deep Learning',
        'Natural Language Processing (NLP)',
        'Computer Vision',
        'Artificial Intelligence',
        'Data Mining',
        'Big Data',
        'Data Visualization',
        'Blockchain',
        'Internet of Things (IoT)',
        'Cybersecurity',
        'Network Administration',
        'System Administration',
        'DevOps',
        'Software Development Life Cycle (SDLC)',
        'Agile Software Development',
        'Object-Oriented Programming (OOP)',
        'Functional Programming',
        'Microservices',
        'RESTful APIs',
        'GraphQL',
        'SOAP',
        'Web Services',
        'Responsive Web Design',
        'Cross-Platform Development',
        'Mobile Development',
        'Web Development',
        'Backend Development',
        'Frontend Development',
        'Full-Stack Development',
        'UI/UX Design',
        'Version Control',
        'Software Architecture',
        'Algorithms',
        'Data Structures',
        'Problem-Solving',
        'Debugging',
        'Code Optimization',
        'Documentation',
        'Technical Writing',
        'Code Review',
        # Additional Programming Languages
        'Rust',
        'Dart',
        'Haskell',
        'Groovy',
        'Clojure',
        'Elixir',
        'Lua',
        'Julia',
        'Assembly Language',

        # Additional Web Frameworks
        'Laravel',
        'Symfony',
        'Ruby on Rails',
        'Ember.js',
        'Svelte',
        'Next.js',

        # Additional Libraries and Frameworks
        'Flask-RESTful',
        'Spring Boot',
        'Hibernate',
        'ASP.NET Core',
        'Tailwind CSS',
        'Bulma',

        # Additional Machine Learning and Data Science Tools
        'XGBoost',
        'LightGBM',
        'CatBoost',
        'Prophet',
        'H2O.ai',

        # Additional Database Systems
        'MariaDB',
        'Oracle',
        'IBM DB2',
        'Redis',
        'Cassandra',
        'Couchbase',
        'Elasticsearch',

        # Additional Cloud Services
        'Firebase',
        'Heroku',
        'DigitalOcean',
        'IBM Cloud',

        # Additional DevOps Tools
        'Ansible',
        'Terraform',
        'Puppet',
        'Vagrant',
        'Chef',

        # Additional Testing Tools
        'Selenium',
        'Appium',
        'Cypress',
        'JUnit',
        'Mockito',
        'RSpec',
        'NUnit',

        # Additional Skills
        'Microfrontend Architecture',
        'Serverless Architecture',
        'CI/CD Pipelines',
        'Infrastructure as Code (IaC)',
        'Container Orchestration',
        'Service Mesh',
        'WebAssembly',
        'Progressive Web Apps (PWA)',
        'Federated Learning',
        'Privacy-Preserving Machine Learning',
        'Robotic Process Automation (RPA)',
        'Immutable Infrastructure',
        'Chaos Engineering',
        'Event-Driven Architecture',
        'Domain-Driven Design (DDD)',
        'Model-View-ViewModel (MVVM) Pattern',
        'Model-View-Controller (MVC) Pattern',
        'Observer Pattern',
        'Command Query Responsibility Segregation (CQRS)',
        'Event Sourcing',
        'Behavior-Driven Development (BDD)',
        'Acceptance Test-Driven Development (ATDD)',
        'Exploratory Testing',
        'Security Testing',
        'Load Testing',
        'Performance Tuning',
        'Capacity Planning',
        'Disaster Recovery Planning',
        'Compliance Management',
        'Data Governance',
        'Data Warehousing',
        'ETL (Extract, Transform, Load)',
        'Master Data Management (MDM)',
        'Data Lakes',
        'Predictive Analytics',
        'Prescriptive Analytics',
        'Descriptive Analytics',
        'A/B Testing',
        'Multivariate Testing',
        'Content Management Systems (CMS)',
        'E-commerce Platforms',
        'Payment Gateway Integration',
        'Search Engine Optimization (SEO)',
        'Conversion Rate Optimization (CRO)',
        'User Behavior Analysis',
        'Customer Relationship Management (CRM)',
        'Supply Chain Management (SCM)',
        'Enterprise Resource Planning (ERP)',
        'Business Process Management (BPM)',
        'Agile Project Management Tools',
        'Data Ethics',
        'Data Privacy Compliance',
        'GDPR Compliance',
        'HIPAA Compliance',
        'SOX Compliance',
        'ISO/IEC 27001 Compliance',
        'PCI DSS Compliance',
        'OAuth',
        'OpenID Connect',
        'JWT Authentication',
        'OAuth2.0',
        'OpenID Connect',
        'SAML',

        # Soft Skills
        'Effective Communication',
        'Teamwork',
        'Problem-Solving',
        'Time Management',
        'Adaptability',
        'Creativity',
        'Leadership',
        'Critical Thinking',
        'Attention to Detail',
        'Empathy',
        'Conflict Resolution',
        'Negotiation Skills',
        'Stress Management',
        'Decision Making',
        'Presentation Skills',
        'Customer Focus',
        'Resilience'
    }

    def extract_text_from_docx(docx_path):
        txt = docx2txt.process(docx_path)
        if txt:
            return txt.replace('\t', ' ')
        return None

    def extract_skills(input_text):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_tokens = nltk.tokenize.word_tokenize(input_text)

        # remove the stop words
        filtered_tokens = [w for w in word_tokens if w.lower() not in stop_words]

        # remove the punctuation
        filtered_tokens = [w for w in filtered_tokens if w.isalpha()]

        # generate bigrams and trigrams (such as artificial intelligence)
        bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))

        # we create a set to keep the results in
        found_skills = set()

        # we search for each token in our skills database
        for token in filtered_tokens:
            if token.lower() in [skill.lower() for skill in SKILLS_DB]:
                found_skills.add(token)

        # we search for each bigram and trigram in our skills database
        for ngram in bigrams_trigrams:
            if ngram.lower() in [skill.lower() for skill in SKILLS_DB]:
                found_skills.add(ngram)

        return found_skills

    def predict_job_profiles(skills):
        # Define dictionaries mapping skills to potential job profiles for each IT profile
        software_dev_profiles = {
            "python": ["Software Developer", "Data Scientist", "Python Developer", "Backend Developer", "AI Engineer",
                       "Research Scientist"],
            "java": ["Software Developer", "Systems Engineer", "Java Developer", "Backend Developer",
                     "Android Developer", "Enterprise Architect"],
            "c++": ["Software Developer", "Game Developer", "Systems Programmer", "Embedded Systems Engineer",
                    "Quantitative Analyst", "Operating Systems Developer"],
            "javascript": ["Web Developer", "Frontend Developer", "Full-Stack Developer", "UI/UX Developer",
                           "Node.js Developer", "React Developer", "Angular Developer", "Vue.js Developer"],
            "html": ["Web Developer", "Frontend Developer", "UI/UX Developer", "Email Developer", "Web Designer",
                     "Content Manager"],
            "css": ["Web Developer", "Frontend Developer", "UI/UX Developer", "Web Designer", "Interactive Designer",
                    "Frontend Architect"],
            "node.js": ["Web Developer", "Backend Developer", "Full-Stack Developer", "Node.js Developer",
                        "API Developer", "JavaScript Engineer"],
            "django": ["Web Developer", "Backend Developer", "Full-Stack Developer", "Django Developer",
                       "Python Developer", "Web Application Developer", "Flask Developer"],
            "sql": ["Database Administrator", "Data Analyst", "SQL Developer", "Business Intelligence Analyst",
                    "Database Developer", "Data Engineer"],
            "data analysis": ["Data Analyst", "Business Analyst", "Data Scientist", "Quantitative Analyst",
                              "Financial Analyst", "Market Research Analyst"],
            "data visualization": ["Data Analyst", "Data Scientist", "Business Intelligence Analyst",
                                   "Data Visualization Developer", "UI/UX Designer", "Information Designer"],
            "networking protocols": ["Network Engineer", "System Administrator", "Network Architect",
                                     "Network Security Engineer", "DevOps Engineer", "Cloud Network Engineer"],
            "network security": ["Network Engineer", "Cybersecurity Analyst", "Information Security Analyst",
                                 "Security Consultant", "Security Architect", "Penetration Tester"],
            "docker": ["DevOps Engineer", "Cloud Engineer", "Infrastructure Engineer", "Site Reliability Engineer",
                       "Containerization Engineer", "Automation Engineer"],
            "kubernetes": ["DevOps Engineer", "Cloud Engineer", "Container Orchestration Engineer", "Cloud Architect",
                           "System Administrator", "Platform Engineer"],
            "continuous integration": ["DevOps Engineer", "Software Engineer", "Release Engineer", "Build Engineer",
                                       "Integration Engineer", "Quality Engineer"],
            "continuous deployment": ["DevOps Engineer", "Software Engineer", "Release Engineer", "Deployment Engineer",
                                      "Pipeline Engineer", "Automation Engineer"],
            "git": ["DevOps Engineer", "Software Developer", "Version Control Engineer", "Collaboration Engineer",
                    "Configuration Manager", "Source Code Manager"],
            "tensorflow": ["Machine Learning Engineer", "Data Scientist", "AI Engineer", "Deep Learning Engineer",
                           "Research Scientist", "Machine Learning Developer", "PyTorch Developer", "Keras Developer"],
            "react native": ["Mobile Developer", "Frontend Developer", "Mobile App Developer", "UI/UX Developer",
                             "JavaScript Engineer", "Cross-Platform Developer", "Flutter Developer"],
            # Add more skills and their corresponding job profiles as needed
        }

        data_science_profiles = {
            "python": ["Data Scientist", "Data Engineer", "Machine Learning Engineer", "AI Engineer",
                       "Research Scientist"],
            "r": ["Data Scientist", "Statistical Analyst", "Data Analyst", "Quantitative Analyst",
                  "Research Scientist"],
            "sql": ["Data Analyst", "Database Administrator", "Data Engineer", "Business Intelligence Analyst"],
            "machine learning": ["Data Scientist", "Machine Learning Engineer", "AI Engineer", "Research Scientist"],
            "deep learning": ["Data Scientist", "Machine Learning Engineer", "AI Engineer", "Research Scientist"],
            "statistics": ["Data Scientist", "Statistical Analyst", "Quantitative Analyst", "Research Scientist"],
            "data visualization": ["Data Scientist", "Data Analyst", "Data Visualization Developer", "UI/UX Designer"],
            "data mining": ["Data Scientist", "Data Analyst", "Data Engineer", "Business Intelligence Analyst"],
            "big data": ["Data Engineer", "Big Data Engineer", "Data Architect", "Data Scientist"],
            "hadoop": ["Big Data Engineer", "Data Engineer", "Data Architect", "Data Scientist"],
            "spark": ["Big Data Engineer", "Data Engineer", "Data Architect", "Data Scientist"],
            "nlp": ["Data Scientist", "NLP Engineer", "AI Engineer", "Research Scientist"],
            "computer vision": ["Computer Vision Engineer", "Machine Learning Engineer", "AI Engineer",
                                "Research Scientist"],
            "predictive modeling": ["Data Scientist", "Machine Learning Engineer", "AI Engineer", "Research Scientist"],
            # Add more skills and their corresponding job profiles as needed
        }

        devops_profiles = {
            "docker": ["DevOps Engineer", "Cloud Engineer", "Infrastructure Engineer", "Site Reliability Engineer",
                       "Containerization Engineer", "Automation Engineer"],
            "kubernetes": ["DevOps Engineer", "Cloud Engineer", "Container Orchestration Engineer", "Cloud Architect",
                           "System Administrator", "Platform Engineer"],
            "jenkins": ["DevOps Engineer", "Build Engineer", "Release Engineer", "Automation Engineer"],
            "ansible": ["DevOps Engineer", "Automation Engineer", "Configuration Management Engineer"],
            "terraform": ["DevOps Engineer", "Cloud Engineer", "Infrastructure Engineer", "Automation Engineer"],
            "git": ["DevOps Engineer", "Version Control Engineer", "Collaboration Engineer", "Configuration Manager"],
            "linux": ["DevOps Engineer", "System Administrator", "Infrastructure Engineer"],
            "bash": ["DevOps Engineer", "System Administrator", "Automation Engineer"],
            "python": ["DevOps Engineer", "Automation Engineer", "Scripting Engineer"],
            # Add more skills and their corresponding job profiles as needed
        }

        cybersecurity_profiles = {
            "network security": ["Cybersecurity Analyst", "Information Security Analyst", "Security Consultant",
                                 "Security Architect", "Penetration Tester"],
            "firewalls": ["Cybersecurity Analyst", "Network Security Engineer", "Security Operations Engineer"],
            "siem": ["Cybersecurity Analyst", "Security Operations Analyst", "SOC Analyst"],
            "ethical hacking": ["Cybersecurity Analyst", "Penetration Tester", "Ethical Hacker"],
            "incident response": ["Cybersecurity Analyst", "Incident Responder", "Security Analyst"],
            "risk management": ["Cybersecurity Analyst", "Risk Analyst", "Security Risk Manager"],
            "vulnerability assessment": ["Cybersecurity Analyst", "Vulnerability Assessor", "Security Analyst"],
            "encryption": ["Cybersecurity Analyst", "Cryptographer", "Security Engineer"],
            "security policies": ["Cybersecurity Analyst", "Security Policy Analyst", "Compliance Analyst"],
            # Add more skills and their corresponding job profiles as needed
        }

        cloud_engineer_profiles = {
            "aws": ["Cloud Engineer", "AWS Cloud Engineer", "Cloud Architect", "Solution Architect"],
            "azure": ["Cloud Engineer", "Azure Cloud Engineer", "Cloud Architect", "Solution Architect"],
            "google cloud": ["Cloud Engineer", "Google Cloud Engineer", "Cloud Architect", "Solution Architect"],
            "terraform": ["Cloud Engineer", "Infrastructure Engineer", "Automation Engineer"],
            "ansible": ["Cloud Engineer", "Infrastructure Engineer", "Automation Engineer"],
            "kubernetes": ["Cloud Engineer", "Kubernetes Engineer", "Containerization Engineer", "DevOps Engineer"],
            "docker": ["Cloud Engineer", "Containerization Engineer", "DevOps Engineer"],
            "linux": ["Cloud Engineer", "System Administrator", "Infrastructure Engineer"],
            "networking": ["Cloud Engineer", "Network Engineer", "Infrastructure Engineer"],
            # Add more skills and their corresponding job profiles as needed
        }
        full_stack_skill_profiles = {
            "javascript": ["Full-Stack Developer", "Frontend Developer", "Web Developer", "UI/UX Developer",
                           "Node.js Developer"],
            "html": ["Full-Stack Developer", "Frontend Developer", "Web Developer", "UI/UX Developer"],
            "css": ["Full-Stack Developer", "Frontend Developer", "Web Developer", "UI/UX Developer"],
            "react": ["Full-Stack Developer", "Frontend Developer", "Web Developer", "UI/UX Developer"],
            "angular": ["Full-Stack Developer", "Frontend Developer", "Web Developer", "UI/UX Developer"],
            "vue.js": ["Full-Stack Developer", "Frontend Developer", "Web Developer", "UI/UX Developer"],
            "node.js": ["Full-Stack Developer", "Backend Developer", "Web Developer", "API Developer"],
            "express.js": ["Full-Stack Developer", "Backend Developer", "Web Developer", "API Developer"],
            "mongodb": ["Full-Stack Developer", "Backend Developer", "Database Developer"],
            "mysql": ["Full-Stack Developer", "Backend Developer", "Database Developer"],
            "python": ["Full-Stack Developer", "Backend Developer", "Web Developer", "Python Developer"],
            "django": ["Full-Stack Developer", "Backend Developer", "Web Developer", "Python Developer"],
            "flask": ["Full-Stack Developer", "Backend Developer", "Web Developer", "Python Developer"],
            "ruby": ["Full-Stack Developer", "Backend Developer", "Web Developer"],
            "ruby on rails": ["Full-Stack Developer", "Backend Developer", "Web Developer"],
            "php": ["Full-Stack Developer", "Backend Developer", "Web Developer"],
            "laravel": ["Full-Stack Developer", "Backend Developer", "Web Developer"],
            "java": ["Full-Stack Developer", "Backend Developer", "Web Developer"],
            "spring boot": ["Full-Stack Developer", "Backend Developer", "Web Developer"],
            # Add more skills and their corresponding job profiles as needed
        }
        database_administrator_profiles = {
            "sql": ["Database Administrator", "SQL Database Administrator", "Database Developer"],
            "oracle": ["Database Administrator", "Oracle Database Administrator", "Database Developer"],
            "mysql": ["Database Administrator", "MySQL Database Administrator", "Database Developer"],
            "postgresql": ["Database Administrator", "PostgreSQL Database Administrator", "Database Developer"],
            "database tuning": ["Database Administrator", "Database Tuning Specialist", "Database Performance Analyst"],
            "backup and recovery": ["Database Administrator", "Backup and Recovery Specialist",
                                    "Disaster Recovery Analyst"],
            "data modeling": ["Database Administrator", "Data Modeling Specialist", "Data Architect"],
            "security management": ["Database Administrator", "Database Security Specialist", "Security Administrator"],
            "replication": ["Database Administrator", "Database Replication Specialist", "Data Replication Engineer"],
            # Add more skills and their corresponding job profiles as needed
        }
        network_engineer_profiles = {
            "cisco": ["Network Engineer", "Cisco Network Engineer", "Network Administrator", "Network Analyst"],
            "routing": ["Network Engineer", "Routing Engineer", "Network Administrator"],
            "switching": ["Network Engineer", "Switching Engineer", "Network Administrator"],
            "firewalls": ["Network Engineer", "Firewall Engineer", "Network Security Engineer"],
            "load balancing": ["Network Engineer", "Load Balancing Engineer", "Network Architect"],
            "network protocols": ["Network Engineer", "Network Protocol Engineer", "Network Analyst"],
            "troubleshooting": ["Network Engineer", "Network Troubleshooter", "Network Support Engineer"],
            "dns": ["Network Engineer", "DNS Engineer", "Network Administrator"],
            "vlan": ["Network Engineer", "VLAN Engineer", "Network Administrator"],
            # Add more skills and their corresponding job profiles as needed
        }

        # Initialize a dictionary to store the count of job profiles
        profile_weights = {}

        # Iterate through each skill and update the profile weights
        for skill in skills:
            for profiles_dict in [software_dev_profiles, data_science_profiles, devops_profiles,
                                  cybersecurity_profiles, cloud_engineer_profiles]:
                for key, value in profiles_dict.items():
                    if skill.lower() == key or skill.lower() in value:
                        for profile in value:
                            if profile in profile_weights:
                                profile_weights[profile] += 1
                            else:
                                profile_weights[profile] = 1

        # Normalize the weights
        total_weight = sum(profile_weights.values())
        if total_weight == 0:
            # If no matching profiles found, return a default value
            return ["Unknown"]
        else:
            normalized_weights = {profile: weight / total_weight for profile, weight in profile_weights.items()}

            # Find the job profiles with the highest weights
            max_weight = max(normalized_weights.values())
            predicted_profiles = [profile for profile, weight in normalized_weights.items() if weight == max_weight]

            return predicted_profiles

    if __name__ == '__main__':
        user_data_file = "D:\pythonProject2\quiz-webapp-flask-main-20240421T172231Z-001\quiz-webapp-flask-main/users_data.txt"
        text = extract_text_from_docx(fname)
        # print("Extracted Text:", text)
        skills = extract_skills(text)
        print("Applicant_Skills:", skills)  # Print extracted skills for debugging

        job_profile = predict_job_profiles(skills)
        print("Job Profile:", job_profile)

        # Regex pattern for phone numbers
        phone_pattern = r'\b(?:\+\d{1,2}\s*)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        phone_no = re.findall(phone_pattern, text)
        print("Candidate_contact:", phone_no)

        # Regex pattern for email
        email_pattern = '[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        email_match = re.findall(email_pattern, text)
        print("Candidate_email:", email_match)

        if email_match:
            email = email_match[0]
            # Append the extracted and predicted information to the user data
            append_user_info(user_data_file, email, personality, job_profile)
        else:
            print("Email not found in the resume. Please provide your email ID manually.")

        write_results_to_file(candidateName, age, skills, job_profile, phone_no, email_match, personality)

        # End Process

        # Email Process
        emailform = "srivastavaapaar15@gmail.com"
        emailpassword = "coesykqhnlscvjke"
        emailto = email_match
        # Subject
        subject = "Shortlisted Personality Prediction Test"

        # message to be sent

        message = "Subject: Congratulations! You've Been Shortlisted for the Quiz Round\n\n" + "Dear " + str(
            candidateName) + "\n\n" + "I trust this email finds you in good spirits. On behalf of our organization, I am delighted to extend our warmest congratulations to you. You have been successfully shortlisted for the quiz round.\n\n" + "\n" + "We hope this email finds you well. We are thrilled to inform you that" + " You are successfully  shortlisted for quiz round." + "\n" + " your personality type is: " + str(
            personality) + "\n" + " Please click on the attached link to complete the MCQ Test," + "\n" + " MCQ Test Link: http://localhost:5000/index" + "\n" + "As per the analysis of your resume, your mentioned skill set is suitable for " + str(
            job_profile) + "." + "\n" + "Thankyou."

        em = EmailMessage()
        em['Form'] = ""
        em['To'] = ""
        em['Subject'] = subject
        em.set_content(message)
        # creates SMTP session
        s = smtplib.SMTP('smtp.gmail.com', 587)

        # start TLS for security
        s.starttls()
        # Authentication
        s.login(emailform, emailpassword)

        # sending the mail
        s.sendmail(emailform, emailto, em.as_string())
        # terminating the session
        s.close()
        print("Email sent successfully")
        #  end process

    # Create a new window for displaying the result
    result = Toplevel(top)
    result.geometry('800x600')
    result.configure(background='white')
    result.title("Result - Personality Prediction")

    # Title
    titleFont = font.Font(family='Arial', size=20, weight='bold')
    Label(result, text="Result - Personality Prediction", foreground='green', bg='white', font=titleFont, pady=10,
          anchor=CENTER).pack(fill=BOTH)

    # Create a text widget to display the output
    output_text = Text(result, wrap=WORD, height=30, width=100, font=("Arial", 12), bg="white", fg="black")
    output_text.pack(side=LEFT, fill=Y, padx=10, pady=10)

    # Create a scrollbar and associate it with the text widget
    scrollbar = Scrollbar(result, command=output_text.yview)
    scrollbar.pack(side=RIGHT, fill=Y)
    output_text.config(yscrollcommand=scrollbar.set)

    terms_mean = """
# Openness:
    People who like to learn new things and enjoy new experiences usually score high in openness. Openness includes traits like being insightful and imaginative and having a wide variety of interests.

# Conscientiousness:
    People that have a high degree of conscientiousness are reliable and prompt. Traits include being organised, methodic, and thorough.

# Extraversion:
    Extraversion traits include being; energetic, talkative, and assertive (sometime seen as outspoken by Introverts). Extraverts get their energy and drive from others, while introverts are self-driven get their drive from within themselves.

# Agreeableness:
    As it perhaps sounds, these individuals are warm, friendly, compassionate and cooperative and traits include being kind, affectionate, and sympathetic. In contrast, people with lower levels of agreeableness may be more distant.

# Neuroticism:
    Neuroticism or Emotional Stability relates to degree of negative emotions. People that score high on neuroticism often experience emotional instability and negative emotions. Characteristics typically include being moody and tense.    
"""

    Label(result, text=terms_mean, foreground='green', bg='white', anchor='w', justify=LEFT).pack(fill=BOTH)

    # Display the output in the text widget
    output_text.insert(END, "Candidate Name: " + candidateName + "\n\n", ("bold", "underline"))
    output_text.insert(END, "Candidate Age: " + str(age) + "\n\n", "heading")
    output_text.insert(END, "Applicant Skills:\n", "heading")
    for skill in skills:
        output_text.insert(END, "- " + skill + "\n", "normal")
    output_text.insert(END, "\nCandidate Contact:\n", "heading")
    for contact in phone_no:
        output_text.insert(END, "- " + contact + "\n", "normal")
    output_text.insert(END, "\nCandidate Email:\n", "heading")
    for email in email_match:
        output_text.insert(END, "- " + email + "\n", "normal")
    output_text.insert(END, "\nJob Profile: " + str(job_profile) + "\n ", "heading")
    output_text.insert(END, "Predicted Personality: " + personality + "", "heading")

    # output_text.insert(END, terms_mean, "normal")

    # Define tags for styling
    output_text.tag_configure("bold", font=("Arial", 12, "bold"))
    output_text.tag_configure("underline", underline=True)
    output_text.tag_configure("heading", font=("Arial", 14, "bold"))
    output_text.tag_configure("subheading", font=("Arial", 12, "italic"))
    output_text.tag_configure("normal", font=("Arial", 12))

    # Quit button
    quitBtn = Button(result, text="Exit", command=result.destroy).pack()

    result.mainloop()


def perdict_person():
    """Predict Personality"""

    # Closing The Previous Window
    root.withdraw()

    # Creating new window
    top = Toplevel()
    top.geometry('800x500')
    top.configure(background='#f2f2f2')
    top.title("Apply For A Job")

    # Title
    titleFont = font.Font(family='Helvetica', size=20, weight='bold')
    lab = Label(top, text="Personalty Prediction Via CV Analysis", foreground='black', font=titleFont, pady=10).pack()

    # Define validation functions
    def validate_name(name):
        if name.replace(" ", "").isalpha():  # Remove spaces and then check for alphabetic characters
            return True
        else:
            messagebox.showerror("Error", "Name should contain only alphabetic characters")
            return False

    def validate_age(age):
        if age.isdigit():
            return True
        else:
            messagebox.showerror("Error", "Age should contain only numerical characters")
            return False

    def validate_personality(personality):
        if personality.isdigit() and 0 < int(personality) <= 10:
            return True
        else:
            messagebox.showerror("Error", "Personality values should be numerical and between 1 and 10")
            return False

    # Define submit function
    def submit():
        cv_path = OpenFile(cv)
        if (validate_name(sName.get()) and validate_age(age.get()) and
                validate_personality(openness.get()) and validate_personality(neuroticism.get()) and
                validate_personality(conscientiousness.get()) and validate_personality(agreeableness.get()) and
                validate_personality(extraversion.get())):
            prediction_result(top, sName, cv_path, (
            gender.get(), age.get(), openness.get(), neuroticism.get(), conscientiousness.get(), agreeableness.get(),
            extraversion.get()))

    l1 = Label(top, text="Applicant Name", foreground='black').place(x=70, y=130)
    l2 = Label(top, text="Age", foreground='black').place(x=70, y=160)
    l3 = Label(top, text="Gender", foreground='black').place(x=70, y=190)
    l4 = Label(top, text="Upload Resume", foreground='black').place(x=70, y=220)
    l5 = Label(top, text="Enjoy New Experience or thing(Openness)", foreground='black').place(x=70, y=250)
    l6 = Label(top, text="How Offen You Feel Negativity(Neuroticism)", foreground='black').place(x=70, y=280)
    l7 = Label(top, text="Wishing to do one's work well and thoroughly(Conscientiousness)", foreground='black').place(
        x=70, y=310)
    l8 = Label(top, text="How much would you like work with your peers(Agreeableness)", foreground='black').place(x=70,
                                                                                                                  y=340)
    l9 = Label(top, text="How outgoing and social interaction you like(Extraversion)", foreground='black').place(x=70,
                                                                                                                 y=370)

    # Define variables to capture user inputs
    sName = Entry(top)
    sName.place(x=450, y=130, width=160)
    age = Entry(top)
    age.place(x=450, y=160, width=160)
    gender = IntVar()
    R1 = Radiobutton(top, text="Male", variable=gender, value=1, padx=7)
    R1.place(x=450, y=190)
    R2 = Radiobutton(top, text="Female", variable=gender, value=0, padx=3)
    R2.place(x=540, y=190)

    cv = Button(top, text="Select File", command=lambda: OpenFile(cv))
    cv.place(x=450, y=220, width=160)

    openness = Entry(top)
    openness.insert(0, '1-10')
    openness.place(x=450, y=250, width=160)
    neuroticism = Entry(top)
    neuroticism.insert(0, '1-10')
    neuroticism.place(x=450, y=280, width=160)
    conscientiousness = Entry(top)
    conscientiousness.insert(0, '1-10')
    conscientiousness.place(x=450, y=310, width=160)
    agreeableness = Entry(top)
    agreeableness.insert(0, '1-10')
    agreeableness.place(x=450, y=340, width=160)
    extraversion = Entry(top)
    extraversion.insert(0, '1-10')
    extraversion.place(x=450, y=370, width=160)

    # Define the submit button
    submitBtn = Button(top, padx=2, pady=0, text="Submit", bd=0, foreground='white', bg='green', font=(12))
    submitBtn.config(command=submit)
    submitBtn.place(x=350, y=400, width=200)

    top.mainloop()


def OpenFile(b4):
    global loc;
    global fname;

    name = filedialog.askopenfilename(initialdir="C:/Users/Batman/Documents/Programming/tkinter/",
                                      filetypes=(("Document", "*.docx*"), ("PDF", "*.pdf*"), ('All files', '*')),
                                      title="Choose a file."
                                      )
    try:
        filename = os.path.basename(name)
        loc = name
        fname = filename
        f = request.files[name]
        f.save(f.filename)
    except:
        filename = name
        loc = name
        fname = filename
    b4.config(text=filename)
    return loc  # return loc variable to perdict function


if __name__ == "__main__":
    model = train_model()
    model.train()

    root = Tk()
    root.geometry('800x500')
    root.configure(background='white')
    root.title("Personality Prediction System")
    titleFont = font.Font(family='Helvetica', size=25, weight='bold')
    homeBtnFont = font.Font(size=12, weight='bold')
    lab = Label(root, text="Personalty Prediction Via CV Analysis System", bg='white', font=titleFont, pady=30).pack()
    b2 = Button(root, padx=4, pady=4, width=30, text="Predict Personality", bg='green', foreground='black', bd=1,
                font=homeBtnFont, command=perdict_person).place(relx=0.5, rely=0.5, anchor=CENTER)
    root.mainloop()

