import requests
from bs4 import BeautifulSoup

from scholia import query

class DTUOrbitScraper:
    def __init__(self):
        self.base_url = "https://orbit.dtu.dk/en/persons/"
        
    def search_person(self, name):
        """Search for the person and get the URL to their profile."""
        search_url = f"{self.base_url}?search={name.replace(' ', '+')}&isCopyPasteSearch=false"
        response = requests.get(search_url)
        
        if response.status_code != 200:
            raise Exception("Failed to fetch search results")
        
        soup = BeautifulSoup(response.text, "html.parser")
        # Find the first profile link (assuming it's the first result)
        profile_link = soup.find("h3", class_="title").find("a", href=True)
        
        if profile_link:
            return profile_link['href']
        else:
            raise Exception("Profile link not found")

    def get_profile_info(self, name):
        """Retrieve profile information given a person's name."""
        full_profile_url = self.search_person(name)
        print(full_profile_url)
        response = requests.get(full_profile_url)
        
        if response.status_code != 200:
            raise Exception("Failed to fetch profile page")
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract profile information
        profile_info = {}
        
        # Get Profile Description
        # Find the <h3> with "Profile" text and then find the next <p> tag that contains the profile information
        profile_header = soup.find("h3", string="Profile")
        profile_section = profile_header.find_next("p") if profile_header else None
        profile_info["Profile_desc"] = profile_section.get_text(strip=True) if profile_section else "None"
        
        # Get Keywords
        keywords_section = soup.find("div", class_="keyword-group")
        if keywords_section:
            keywords = [keyword.get_text(strip=True) for keyword in keywords_section.find_all("li", class_="userdefined-keyword")]
            profile_info["Keywords"] = keywords
        else:
            profile_info["Keywords"] = []

        # Get Fingerprint (Concepts, Thesauri, Values)
        fingerprints = []
        fingerprint_section = soup.find("div", class_="person-top-concepts")
        if fingerprint_section:
            fingerprint_items = fingerprint_section.find_all("li", class_="concept-badge-large-container")
            for item in fingerprint_items:
                concept = item.find("span", class_="concept").get_text(strip=True) if item.find("span", class_="concept") else "N/A"
                thesauri = item.find("span", class_="thesauri").get_text(strip=True) if item.find("span", class_="thesauri") else "N/A"
                value = item.find("span", class_="value sr-only").get_text(strip=True) if item.find("span", class_="value sr-only") else "N/A"
                fingerprints.append({
                    "Concept": concept,
                    "Thesauri": thesauri,
                    "Value": value
                })
        profile_info["Fingerprint"] = fingerprints

        # Get ORCID
        orcid_section = soup.find("div", class_="rendering_person_personorcidrendererportal")
        if orcid_section:
            orcid_link = orcid_section.find("a", href=True)
            profile_info["ORCID"] = orcid_link["href"] if orcid_link else "Not found"

            profile_info["QS"] = query.orcid_to_qs(orcid_link["href"].split("/")[-1]) if orcid_link else "Not found"
        else:
            profile_info["ORCID"] = "Not found"
        return profile_info


if __name__ == "__main__":
    scraper = DTUOrbitScraper()
    profile_info = scraper.get_profile_info("Sune Lehmann")
    print(profile_info)