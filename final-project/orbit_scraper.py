import requests
from bs4 import BeautifulSoup

from scholia import query as scholia_query
from SPARQLWrapper import SPARQLWrapper, JSON

class DTUOrbitScraper:
    def __init__(self):
        self.base_url = "https://orbit.dtu.dk/en/persons/"
        self.endpoint_url = "https://query.wikidata.org/sparql"

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

    def get_topic_info(self, topic_url):
        """Scrape the description for a topic from its Wikidata page."""
        response = requests.get(topic_url)
        if response.status_code != 200:
            return "Description not found"
        
        soup = BeautifulSoup(response.text, "html.parser")
        description = soup.find("div", class_="wikibase-entitytermsview-heading-description")
        return description.text.strip() if description else "Description not found"

    def get_scholia_topics(self, qs):
        """Get topics and scores from Scholia using SPARQL."""
        query = f"""PREFIX target: <http://www.wikidata.org/entity/{qs}>
        SELECT ?score ?topic ?topicLabel
        WITH {{
            SELECT (SUM(?score_) AS ?score) ?topic WHERE {{
                {{ target: wdt:P101 ?topic . BIND(20 AS ?score_) }}
                UNION {{ SELECT (3 AS ?score_) ?topic WHERE {{ ?work wdt:P50 target: ; wdt:P921 ?topic . }} }}
                UNION {{ SELECT (1 AS ?score_) ?topic WHERE {{ ?work wdt:P50 target: . ?citing_work wdt:P2860 ?work . ?citing_work wdt:P921 ?topic . }} }}
            }} GROUP BY ?topic
        }} AS %results 
        WHERE {{
            INCLUDE %results
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        ORDER BY DESC(?score)
        LIMIT 200"""
        
        sparql = SPARQLWrapper(self.endpoint_url)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        
        topics = [] 

        for result in results["results"]["bindings"]:
            topic_url = result["topic"]["value"]
            topic_label = result["topicLabel"]["value"]
            score = int(result["score"]["value"])
            #info = self.get_topic_info(topic_url)
            #topics[topic_label] = {"score": score, "info": info}
            topics.append({"topic":topic_label, "score": score, "topic_url": topic_url})
        return topics

    def get_profile_info(self, name):
        """Retrieve profile information given a person's name."""
        full_profile_url = self.search_person(name)
        response = requests.get(full_profile_url)
        
        if response.status_code != 200:
            raise Exception("Failed to fetch profile page")
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract profile information
        profile_info = {}
        
        # Get Profile Description
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

        # Extract ORCID
        orcid_section = soup.find("div", class_="rendering_person_personorcidrendererportal")
        if orcid_section:
            orcid_link = orcid_section.find("a", href=True)
            profile_info["ORCID"] = orcid_link["href"] if orcid_link else "Not found"
            if orcid_link:
                orcid_id = orcid_link["href"].split("/")[-1]
                profile_info["QS"] = scholia_query.orcid_to_qs(orcid_id)
                # Retrieve Scholia topics if QS exists
                if len(profile_info["QS"]) == 1:
                    profile_info["scholia_topics"] = self.get_scholia_topics(profile_info["QS"][0])
                else:
                    profile_info["scholia_topics"] = {}
        else:
            profile_info["ORCID"] = "Not found"
            profile_info["QS"] = "Not found"
            profile_info["scholia_topics"] = {}

        return profile_info

if __name__ == "__main__":
    scraper = DTUOrbitScraper()
    profile_info = scraper.get_profile_info("Sune Lehmann")
    print(profile_info)