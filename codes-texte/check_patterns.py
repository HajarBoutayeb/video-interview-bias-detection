import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import difflib

class BiasPatternChecker:
    def __init__(self, transcript_folder: str):
        self.transcript_folder = transcript_folder
        self.patterns = self._load_patterns_from_excel()
        self.detections = []
        
    def _load_patterns_from_excel(self) -> List[Tuple]:
        """Charge les patterns de biais depuis le fichier Excel"""
        excel_path = "C:/Users/ADmiN/Desktop/video_project/bias_patterns.xlsx"
        
        try:
            df = pd.read_excel(excel_path)
            patterns = []
            
            for _, row in df.iterrows():
                pattern = row['Pattern']
                bias_type = row['Bias Type']
                source = row['Source']
                patterns.append((pattern, bias_type, source))
            
            print(f"✅ {len(patterns)} patterns chargés depuis {excel_path}")
            return patterns
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du fichier Excel: {e}")
            return []
    
    def get_severity_level(self, bias_type: str) -> str:
        """Détermine le niveau de gravité basé sur le type de biais"""
        positive_practices = [
            "Procedural Fairness",
            "Anti-Bias Policy",
            "Self-Reflection Bias Acknowledgment"
        ]
        
        if bias_type in positive_practices:
            return "POSITIVE"
        
        high_severity = [
            "Direct Discrimination", 
            "Race / Ethnicity Bias", 
            "Nationality/Ethnicity Bias",
            "Gender Bias",
            "Disability Bias"
        ]
        
        medium_severity = [
            "Stereotype Bias",
            "Cultural Bias", 
            "Language Bias",
            "Age Bias",
            "Education Bias",
            "Nepotism / Cronyism Bias",
            "Seniority Bias"
        ]
        
        if bias_type in high_severity:
            return "HIGH"
        elif bias_type in medium_severity:
            return "MEDIUM"
        else:
            return "LOW"
    
    def detect_patterns_in_text(self, text: str, filename: str) -> List[Dict]:
        """Détecte tous les patterns dans un texte donné avec gestion des doublons et similarité améliorée"""
        detections = []
        seen_matches = set()  # Pour éviter les doublons
        
        for pattern, bias_type, source in self.patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                matched_text = match.group().strip()
                
                # Vérifier si ce texte a déjà été détecté
                match_key = (matched_text.lower(), bias_type)
                if match_key in seen_matches:
                    continue
                seen_matches.add(match_key)
                
                # Nettoyage pour le calcul de similarité
                clean_pattern = re.sub(r'\(\?i\)|\\b|[()]', '', pattern.lower())
                clean_matched = matched_text.lower()
                
                # Calcul de similarité plus précis
                similarity = difflib.SequenceMatcher(
                    None, 
                    clean_matched, 
                    clean_pattern
                ).ratio()
                
                similarity_score = round(similarity * 100, 2)
                
                # Ajustement pour les correspondances exactes
                if re.fullmatch(pattern, matched_text, re.IGNORECASE):
                    similarity_score = 100.0
                
                # Déterminer la gravité
                severity = self.get_severity_level(bias_type)
                
                # Ajouter le contexte
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].replace('\n', ' ').strip()
                
                detection = {
                    'filename': filename,
                    'bias_type': bias_type,
                    'severity': severity,
                    'pattern': pattern,
                    'matched_text': matched_text,
                    'source': source,
                    'similarity': f"{similarity_score}%",
                    'context': context,
                    'line_number': text.count('\n', 0, match.start()) + 1
                }
                
                detections.append(detection)
        
        return detections
    
    def process_file(self, filepath: str) -> List[Dict]:
        """Traite un fichier transcript individuel"""
        try:
            # Essayer différents encodages
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"❌ Impossible de lire le fichier: {filepath}")
                return []
            
            filename = os.path.basename(filepath)
            detections = self.detect_patterns_in_text(content, filename)
            
            return detections
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {filepath}: {e}")
            return []
    
    def check_all_transcripts(self) -> List[Dict]:
        """Traite tous les transcripts dans le dossier"""
        all_detections = []
        transcript_path = Path(self.transcript_folder)
        
        if not transcript_path.exists():
            print(f"❌ Le dossier {self.transcript_folder} n'existe pas.")
            return []
        
        # Extensions de fichiers supportées
        file_extensions = ['*.txt', '*.json', '*.csv', '*.tsv']
        files_processed = 0
        
        print(f"🔍 Recherche de fichiers dans: {transcript_path}")
        
        for extension in file_extensions:
            for filepath in transcript_path.glob(extension):
                print(f"📄 Traitement: {filepath.name}")
                detections = self.process_file(str(filepath))
                all_detections.extend(detections)
                files_processed += 1
                
                if detections:
                    print(f"   ✅ {len(detections)} détections trouvées")
                else:
                    print(f"   ℹ️  Aucun biais détecté")
        
        print(f"\n📊 Résumé:")
        print(f"   • {files_processed} fichiers traités")
        print(f"   • {len(all_detections)} détections totales")
        
        self.detections = all_detections
        return all_detections
    
    def generate_detailed_report(self, output_file: str = "text labels2.xlsx"):
        """Génère un rapport Excel détaillé"""
        if not self.detections:
            print("❌ Aucune détection à reporter. Exécutez d'abord check_all_transcripts().")
            return
        
        df = pd.DataFrame(self.detections)
        
        # Ordonner les colonnes
        columns_order = [
            'filename', 
            'line_number',
            'bias_type', 
            'severity',
            'matched_text',
            'context',
            'similarity',
            'pattern',
            'source'
        ]
        df = df[columns_order]
        
        # Statistiques par type de biais
        bias_stats = df['bias_type'].value_counts().reset_index()
        bias_stats.columns = ['bias_type', 'count']
        bias_stats = bias_stats.sort_values('count', ascending=False)
        
        # Statistiques par niveau de gravité
        severity_stats = df['severity'].value_counts().reset_index()
        severity_stats.columns = ['severity', 'count']
        
        # Statistiques par fichier
        file_stats = df.groupby('filename').agg({
            'bias_type': 'count',
            'severity': lambda x: x.value_counts().to_dict()
        }).reset_index()
        file_stats.columns = ['filename', 'total_detections', 'severity_breakdown']
        
        # Top patterns les plus fréquents
        pattern_stats = df.groupby(['pattern', 'bias_type', 'severity']).size().reset_index(name='frequency')
        pattern_stats = pattern_stats.sort_values('frequency', ascending=False)
        
        # Détections par source vidéo
        source_stats = df.groupby(['source', 'bias_type', 'severity']).size().reset_index(name='count')
        
        # Sauvegarder en Excel
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # Onglet principal avec toutes les détections
            df.to_excel(writer, sheet_name='🔍 Toutes_Détections', index=False)
            
            # Statistiques
            bias_stats.to_excel(writer, sheet_name='📊 Stats_Biais', index=False)
            severity_stats.to_excel(writer, sheet_name='⚠️ Stats_Gravité', index=False)
            file_stats.to_excel(writer, sheet_name='📁 Stats_Fichiers', index=False)
            pattern_stats.to_excel(writer, sheet_name='🔍 Top_Patterns', index=False)
            source_stats.to_excel(writer, sheet_name='🎬 Stats_Sources', index=False)
            
            # Formatage conditionnel
            workbook = writer.book
            worksheet = writer.sheets['🔍 Toutes_Détections']
            
            # Formats pour la gravité
            format_high = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
            format_medium = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
            format_low = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
            format_positive = workbook.add_format({'bg_color': '#D9EAD3', 'font_color': '#0A3B00'})
            
            # Appliquer le formatage conditionnel pour la colonne Severity (D)
            worksheet.conditional_format(
                'D2:D1000',
                {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': 'HIGH',
                    'format': format_high
                }
            )
            worksheet.conditional_format(
                'D2:D1000',
                {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': 'MEDIUM',
                    'format': format_medium
                }
            )
            worksheet.conditional_format(
                'D2:D1000',
                {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': 'LOW',
                    'format': format_low
                }
            )
            worksheet.conditional_format(
                'D2:D1000',
                {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': 'POSITIVE',
                    'format': format_positive
                }
            )
            
            # Formatage pour la similarité (G)
            worksheet.conditional_format(
                'G2:G1000',
                {
                    'type': 'cell',
                    'criteria': '>=',
                    'value': 80,
                    'format': format_low
                }
            )
            worksheet.conditional_format(
                'G2:G1000',
                {
                    'type': 'cell',
                    'criteria': 'between',
                    'minimum': 50,
                    'maximum': 79,
                    'format': format_medium
                }
            )
            worksheet.conditional_format(
                'G2:G1000',
                {
                    'type': 'cell',
                    'criteria': '<',
                    'value': 50,
                    'format': format_high
                }
            )
        
        print(f"✅ Rapport détaillé généré: {output_file}")
        return output_file
    
    def print_summary(self):
        """Affiche un résumé détaillé des résultats"""
        if not self.detections:
            print("❌ Aucune détection disponible.")
            return
        
        df = pd.DataFrame(self.detections)
        
        print("\n" + "="*80)
        print("🎯 RAPPORT DE DÉTECTION DE BIAIS DANS LES ENTRETIENS")
        print("="*80)
        
        print(f"\n📈 STATISTIQUES GÉNÉRALES:")
        print(f"   • Total détections: {len(self.detections)}")
        print(f"   • Fichiers analysés: {df['filename'].nunique()}")
        print(f"   • Types de biais différents: {df['bias_type'].nunique()}")
        
        print(f"\n⚠️ RÉPARTITION PAR GRAVITÉ:")
        severity_counts = df['severity'].value_counts()
        for severity, count in severity_counts.items():
            print(f"   • {severity}: {count}")
        
        print(f"\n📊 TOP 10 TYPES DE BIAIS:")
        bias_counts = df['bias_type'].value_counts().head(10)
        for i, (bias_type, count) in enumerate(bias_counts.items(), 1):
            severity = self.get_severity_level(bias_type)
            print(f"   {i:2d}. {bias_type} ({severity}): {count}")
        
        print(f"\n📁 FICHIERS LES PLUS PROBLÉMATIQUES:")
        file_counts = df['filename'].value_counts().head(5)
        for filename, count in file_counts.items():
            print(f"   • {filename}: {count} détections")
        
        print(f"\n🎬 RÉPARTITION PAR SOURCE:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"   • {source}: {count}")
        
        # Afficher quelques exemples
        examples = df.head(3)
        if not examples.empty:
            print(f"\n📝 EXEMPLES DE DÉTECTIONS:")
            for i, (_, detection) in enumerate(examples.iterrows(), 1):
                print(f"\n   {i}. Fichier: {detection['filename']} (Ligne {detection['line_number']})")
                print(f"      Type: {detection['bias_type']} ({detection['severity']})")
                print(f"      Similarité: {detection['similarity']}")
                print(f"      Contexte: \"...{detection['context']}...\"")
                print(f"      Texte exact: \"{detection['matched_text']}\"")


def main():
    """Fonction principale"""
    print("🚀 Démarrage de l'analyse des biais...")
    
    # Configuration
    transcript_folder = "C:/Users/ADmiN/Desktop/video_project/Interview_Transcripts_For_Bias"
    
    # Initialiser le checker
    checker = BiasPatternChecker(transcript_folder)
    
    # Analyser tous les transcripts
    detections = checker.check_all_transcripts()
    
    if detections:
        # Afficher le résumé
        checker.print_summary()
        
        # Générer le rapport
        report_file = checker.generate_detailed_report()
        
        print(f"\n✅ Analyse terminée!")
        print(f"📊 {len(detections)} détections de biais trouvées")
        print(f"📄 Rapport détaillé: {report_file}")
    else:
        print("\n✅ Analyse terminée - Aucun biais détecté dans les fichiers.")

if __name__ == "__main__":
    main()