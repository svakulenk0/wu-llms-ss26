import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import re
import sys
import time
import csv

# Re-use logic from fetchData.py implicitly or copy for independence
LAW_MAP = {
    "EStG": "Einkommensteuergesetz",
    "UStG": "Umsatzsteuergesetz",
    "KStG": "Körperschaftsteuergesetz",
    "BAO": "Bundesabgabenordnung",
    "ASVG": "Allgemeines Sozialversicherungsgesetz",
    "GSVG": "Gewerbliches Sozialversicherungsgesetz",
    "PStG": "Personenstandsgesetz",
    "ABGB": "Allgemeines bürgerliches Gesetzbuch",
    "GebG": "Gebührengesetz",
    "BewG": "Bewertungsgesetz",
    "FLAG": "Familienlastenausgleichsgesetz",
    "NoVAG": "Normverbrauchsabgabegesetz",
    "GrEStG": "Grunderwerbsteuergesetz",
    "VersStG": "Versicherungssteuergesetz",
    "ZollG": "Zollgesetz",
    "UmgrStG": "Umgründungssteuergesetz",
    "KommStG": "Kommunalsteuergesetz",
    "GmbHG": "GmbH-Gesetz",
    "ErbStG": "Erbschafts- und Schenkungssteuergesetz",
    "InvPrG": "Investitionsprämiengesetz",
    "NEHG": "Nationales Emissionszertifikatehandelsgesetz",
    "DiStG": "Digitalsteuergesetz",
    "RKS-V": "Registrierkassensicherheitsverordnung",
    "Sachbezugswerte-V": "Sachbezugswerteverordnung",
    "LStG": "Landessteuergesetze"
}

def expand_law_reference(law):
    law = law.strip()
    for short, full in LAW_MAP.items():
        if law.startswith(short):
            law = law.replace(short, full, 1)
    law = law.replace("Abs.", "Absatz").replace("Z ", "Ziffer ")
    return law

def extract_text_from_url(url):
    try:
        header = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=header, timeout=10)
        response.raise_for_status()
        if url.lower().endswith('.pdf') or 'application/pdf' in response.headers.get('Content-Type', ''):
            from PyPDF2 import PdfReader
            pdf_file = BytesIO(response.content)
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
            return text.strip()
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Prioritize the main legal text container
            container = soup.find(id="MainContent_DocumentRepeater_BundesnormenDocumentData_0_TextContainer_0")
            if not container:
                # Fallback to general content area
                container = soup.find('div', class_='documentContent')

            if container:
                text = container.get_text(separator='\n', strip=True)
            else:
                # Absolute fallback: remove noise and get all text
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()
                text = soup.get_text(separator='\n', strip=True)
                
            return text
    except Exception as e:
        return f"Error: {e}"

def extract_relevant_paragraphs(full_text, law_ref):
    if not isinstance(full_text, str) or "Error:" in full_text:
        return ""
    
    full_text = full_text.replace('\xa0', ' ')
    
    # Header stripping
    start_patterns = [r'§\s*\d+', r'Artikel\s+[IVX\d]+', r'Art\.\s+[IVX\d]+', r'Der Nationalrat hat beschlossen:']
    first_occurrence = len(full_text)
    for pattern in start_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match and match.start() < first_occurrence:
            first_occurrence = match.start()
    if first_occurrence < len(full_text):
        full_text = full_text[first_occurrence:]

    parts = law_ref.split()
    num = None
    for i, part in enumerate(parts):
        if re.match(r'(§|§§|Artikel|Art\.|Art)', part, re.IGNORECASE):
            if i + 1 < len(parts):
                num_match = re.search(r'([A-Z\d]+[a-z]?)', parts[i+1], re.IGNORECASE)
                if num_match:
                    num = num_match.group(1)
                    break
    if not num:
        return ""

    pattern = re.compile(
        rf"(?:§|Artikel|Art\.)[\s\.]*{num}.*?(?=\n\s*(?:§|Artikel|Art\.)[\s\.]*\s*[A-Z\d]+|$)",
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(full_text)
    if match:
        return match.group(0).strip()
    
    # Fallback
    paragraphs = re.split(r'\n\s*\n', full_text)
    clean = [p.strip() for p in paragraphs if len(p.strip()) > 100 and "RIS Dokument" not in p]
    import random
    if clean:
        return "\n\n".join(random.sample(clean, min(len(clean), 2)))
    return ""

def search_ris_link(law_ref):
    """Constructs a direct RIS search URL and finds the document link."""
    # Clean input: remove stray quotes and extra space
    law_ref = law_ref.replace('"', '').strip()
    
    # 1. Extract Paragraph Number
    paragraph_num = None
    num_match = re.search(r'(?:§+|Artikel|Art\.?)\s*([A-Z\d]+)', law_ref, re.IGNORECASE)
    if num_match:
        paragraph_num = num_match.group(1)
    
    if not paragraph_num:
        print(f"  Could not find paragraph number in: {law_ref}")
        return None

    # 2. Extract Law Title
    # Remove paragraph markers, numbers, and common subsection abbreviations
    clean_ref = re.sub(r'(?:§+|Artikel|Art\.?)\s*[A-Z\d]+[a-z]?', '', law_ref, flags=re.IGNORECASE)
    clean_ref = re.sub(r'(?:Abs\.|Absatz|Z\.|Ziffer|lit\.|Rz)\s*[A-Z\d/\-]+', '', clean_ref, flags=re.IGNORECASE)
    law_title = ' '.join(clean_ref.replace(';', '').replace(',', '').split()) # Normalize internal whitespace
    
    # If law title is empty or just a year, try to get it from LAW_MAP
    if len(law_title) < 3 or law_title.isdigit():
        for short, full in LAW_MAP.items():
            if short in law_ref:
                law_title = short
                break
    
    import urllib.parse
    encoded_title = urllib.parse.quote(law_title)
    
    # Use today's date for FassungVom if needed
    from datetime import datetime
    today = datetime.now().strftime("%d.%m.%Y")

    # Construct the full RIS search URL based on user example
    # Note: Use 'Ergebnis.wxe' instead of 'Suchen.wxe' to get results directly
    ris_search_url = (
        f"https://www.ris.bka.gv.at/Ergebnis.wxe?"
        f"Abfrage=Bundesnormen&"
        f"Kundmachungsorgan=&Index=&"
        f"Titel={encoded_title}&"
        f"Gesetzesnummer=&VonArtikel=&BisArtikel=&"
        f"VonParagraf={paragraph_num}&BisParagraf={paragraph_num}&"
        f"VonAnlage=&BisAnlage=&Typ=&Kundmachungsnummer=&Unterzeichnungsdatum=&"
        f"FassungVom={today}&"
        f"VonInkrafttretedatum=&BisInkrafttretedatum=&"
        f"VonAusserkrafttretedatum=&BisAusserkrafttretedatum=&"
        f"NormabschnittnummerKombination=Und&"
        f"ResultPageSize=100&"
        f"Suchworte=&ShowEmptySearchResultMessage=true&"
        f"SkipToDocumentPage=True"
    )
    
    print(f"  Searching RIS for Title='{law_title}', Paragraf='{paragraph_num}'")
    print(f"  URL: {ris_search_url}")
    
    try:
        header = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(ris_search_url, headers=header, allow_redirects=True, timeout=15)
        
        # If it redirected directly to a document
        if "NormDokument.wxe" in response.url or "/eli/" in response.url:
            return response.url
            
        # Parse the result page
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for the result list
        # RIS search results are often in a list with specific IDs or classes
        # We look for links that contain NormDokument.wxe or Eli
        potential_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if "/eli/" in href or "NormDokument.wxe" in href:
                if href.startswith('/'):
                    href = "https://www.ris.bka.gv.at" + href
                potential_links.append(href)
        
        if potential_links:
            # Usually the first one is the most relevant
            return potential_links[0]
                
        return None
    except Exception as e:
        print(f"  RIS Search error: {e}")
        return None

def process_paragraphs(input_file, output_file):
    print(f"Reading {input_file}...")
    
    # Read file line by line for maximum robustness
    all_refs = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split by semicolon and clean
                parts = [p.strip() for p in line.split(';') if p.strip()]
                for p in parts:
                    if p.lower() != 'nan' and len(p) > 2:
                        all_refs.append(p)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not all_refs:
        print("No valid law references found in the input file.")
        return

    # Deduplicate while preserving order
    unique_refs = list(dict.fromkeys(all_refs))
    
    print(f"Found {len(all_refs)} total references.")
    print(f"Processing {len(unique_refs)} unique references.\n")
    
    results = []
    try:
        for i, ref in enumerate(unique_refs):
            print(f"[{i+1}/{len(unique_refs)}] Processing: {ref}")
            expanded = expand_law_reference(ref)
            url = search_ris_link(expanded)
            
            text = ""
            if url:
                print(f"  Found URL: {url}")
                full_text = extract_text_from_url(url)
                text = extract_relevant_paragraphs(full_text, expanded)
                time.sleep(1) # Be nice to RIS
            else:
                print("  No RIS link found.")
            
            results.append({
                'Original Reference': ref,
                'Full Reference': expanded,
                'RIS URL': url,
                'Extracted Text': text
            })
            
            # Save incrementally
            if (i + 1) % 10 == 0:
                pd.DataFrame(results).to_csv(output_file, index=False, sep=';', quoting=csv.QUOTE_ALL)
                print(f"  (Progress saved to {output_file})")

            # Rate limiting for Google - increase slightly to avoid blocks
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n\n🛑 Script interrupted by user. Saving partial results...")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}. Saving partial results...")

    if results:
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_file, index=False, sep=';', quoting=csv.QUOTE_ALL)
        print(f"\n✅ Done! Final results saved to {output_file}")
    else:
        print("\nNo results to save.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetchFromRIS.py <input_csv> [output_csv]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "training_data.csv"
    
    process_paragraphs(input_csv, output_csv)
