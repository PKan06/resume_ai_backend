# server/rag/profile_extractor.py
import re
from typing import List, Dict, Any


class ProfileExtractor:

    def __init__(self, documents):

        self.documents = documents
        self.person_name = None
        self.person_profile = {}

    def extract(self):

        full_text = " ".join([doc.page_content for doc in self.documents])

        self.person_name = self._extract_name(full_text)

        self.person_profile = {
            'name': self.person_name,
            'skills': self._extract_skills(full_text),
            'experience': self._extract_experience(full_text),
            'education': self._extract_education(full_text),
            'achievements': self._extract_achievements(full_text),
            'contact_info': self._extract_contact_info(full_text),
            'summary': self._extract_summary(full_text)
        }

        return self.person_name, self.person_profile
    
    def _extract_name(self, text: str) -> str:
        name_patterns = [
            # ALL CAPS: "NEBULAA KHAN"
            r'^([A-Z]{2,}(?:\s+[A-Z]{2,})+)',
            # Title Case at start: "Nebulaa Khan"
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*){1,3})',
            # After "Name:" label
            r'Name[:\s]+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,2})',
            # After "Resume of" / "CV of"
            r'(?:Resume of|CV of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]

        for pattern in name_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                name = match.group(1).strip().title()  # normalize to Title Case
                if 2 <= len(name.split()) <= 4:
                    return name

        return "Professional Candidate"
    
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical and soft skills"""
        skill_pattern = r'\b(?:Python|Java|JavaScript|TypeScript|React|Angular|Vue|Node\.js|Express|Django|Flask|FastAPI|Spring|SQL|MySQL|PostgreSQL|MongoDB|Redis|AWS|Azure|GCP|Docker|Kubernetes|Git|Linux|Windows|Machine Learning|AI|Data Science|Analytics|Tableau|Power BI|Pandas|NumPy|TensorFlow|PyTorch|Scikit-learn|REST|GraphQL|API|Microservices|DevOps|CI/CD|Jenkins|GitHub|Agile|Scrum|Leadership|Management|Communication|Problem Solving|Team Work|Project Management)\b'

        skills = set(re.findall(skill_pattern, text, re.IGNORECASE))
        return sorted(list(skills))

    def _extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract work experience"""
        experience_years = re.findall(
            r'\b(\d+)(?:\+)?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)\b', text, re.IGNORECASE)
        companies = re.findall(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s+(?:Inc|Corp|Ltd|Company|Technologies|Solutions|Systems|Pvt|Limited)\b', text)

        return {
            'years': max([int(year) for year in experience_years], default=0),
            'companies': list(set(companies)),
            'roles': re.findall(r'\b(?:Software Engineer|Developer|Analyst|Manager|Lead|Senior|Junior|Intern)\b', text, re.IGNORECASE)
        }

    def _extract_education(self, text: str) -> Dict[str, Any]:
        """Extract education information"""
        degrees = re.findall(
            r'\b(?:Bachelor|Master|PhD|B\.Tech|M\.Tech|B\.E|M\.E|BCA|MCA|MBA|B\.Sc|M\.Sc)(?:\s+of\s+[A-Za-z\s]+)?\b', text, re.IGNORECASE)
        universities = re.findall(
            r'\b(?:University|College|Institute|School)\s+of\s+[A-Za-z\s]+|\b[A-Za-z\s]+\s+(?:University|College|Institute)\b', text, re.IGNORECASE)

        return {
            'degrees': list(set(degrees)),
            'institutions': list(set(universities)),
            'fields': re.findall(r'\b(?:Computer Science|Engineering|Technology|Business|Management|Science)\b', text, re.IGNORECASE)
        }

    def _extract_achievements(self, text: str) -> List[str]:
        """Extract achievements"""
        achievement_patterns = [
            r'(?:achieved|accomplished|awarded|recognized|certified|published|led|increased|improved|reduced|developed|built|created)[^.!?]*[.!?]',
            r'(?:winner|champion|top performer|best|excellent|outstanding)[^.!?]*[.!?]'
        ]

        achievements = []
        for pattern in achievement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            achievements.extend([match.strip() for match in matches])

        return achievements[:10]

    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information"""
        email = re.search(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        phone = re.search(r'[\+]?[1-9]?[0-9]{7,15}', text)
        linkedin = re.search(
            r'(?:linkedin\.com/in/|linkedin\.com/profile/)([A-Za-z0-9-]+)', text, re.IGNORECASE)
        github = re.search(
            r'(?:github\.com/)([A-Za-z0-9-]+)', text, re.IGNORECASE)

        return {
            'email': email.group(0) if email else '',
            'phone': phone.group(0) if phone else '',
            'linkedin': linkedin.group(1) if linkedin else '',
            'github': github.group(1) if github else ''
        }

    def _extract_summary(self, text: str) -> str:
        """Extract professional summary"""
        summary_patterns = [
            r'(?:SUMMARY|OBJECTIVE|PROFILE|ABOUT)[:\s]+((?:.|\n)*?)(?:\n\n|\n[A-Z])',
            r'^((?:.|\n){100,300})'
        ]

        for pattern in summary_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).strip()[:300]

        return f"Experienced professional with expertise in {', '.join(self.person_profile.get('skills', [])[:5])}"

    def classify_question(self, question: str) -> str:

        question_lower = question.lower()

        professional_keywords = [
            'experience', 'skills', 'qualification', 'education', 'work',
            'job', 'career', 'resume', 'cv', 'background', 'expertise',
            'achievement', 'project', 'degree', 'university', 'company',
            'position', 'role', 'responsibility', 'technology',
            'programming', 'certification', 'award', 'accomplishment',
            self.person_name.lower()
        ]

        if any(keyword in question_lower for keyword in professional_keywords):
            return 'professional'

        elif any(greeting in question_lower for greeting in [
            'hello', 'hi', 'hey', 'good morning', 'good evening'
        ]):
            return 'greeting'

        else:
            return 'general'