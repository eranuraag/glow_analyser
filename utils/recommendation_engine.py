import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .product_database import load_product_database
from urllib.parse import urlencode, quote_plus

class SmartRecommendationEngine:
    def __init__(self):
        self.products_df = load_product_database()
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.content_similarity_matrix = None
        self.setup_content_based_filtering()
    
    def setup_content_based_filtering(self):
        """Setup content-based filtering using TF-IDF"""
        # Combine ingredients and skin concerns for content analysis
        content_features = self.products_df['ingredients'] + ' ' + self.products_df['skin_concerns']
        
        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_features)
        
        # Calculate cosine similarity
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
    
    def analyze_skin_needs(self, analysis_results):
        """Analyze skin analysis results and determine product needs"""
        skin_concerns = []
        priority_scores = {}
        
        # UV Damage Analysis
        uv_damage = analysis_results.get('uvDamage', 0)
        if uv_damage > 60:
            skin_concerns.extend(['UV damage', 'sun spots', 'aging'])
            priority_scores['UV damage'] = 3  # High priority
        elif uv_damage > 30:
            skin_concerns.extend(['UV protection', 'antioxidant'])
            priority_scores['UV protection'] = 2  # Medium priority
        
        # Acne Analysis
        acne_severity = analysis_results.get('acneSeverity', 'Minimal')
        if acne_severity in ['Severe', 'Moderate']:
            skin_concerns.extend(['acne', 'blackheads', 'large pores', 'oily skin'])
            priority_scores['acne'] = 3
        elif acne_severity == 'Mild':
            skin_concerns.extend(['blemishes', 'pores'])
            priority_scores['blemishes'] = 2
        
        # Aging/Wrinkle Analysis
        wrinkle_score = analysis_results.get('wrinkleScore', 0)
        if wrinkle_score > 50:
            skin_concerns.extend(['wrinkles', 'fine lines', 'aging', 'firmness'])
            priority_scores['aging'] = 3
        elif wrinkle_score > 25:
            skin_concerns.extend(['fine lines', 'prevention'])
            priority_scores['prevention'] = 2
        
        # Hydration Analysis
        hydration_score = analysis_results.get('hydrationScore', 100)
        if hydration_score < 30:
            skin_concerns.extend(['dryness', 'dehydration', 'rough texture'])
            priority_scores['dryness'] = 3
        elif hydration_score < 60:
            skin_concerns.extend(['hydration', 'moisture'])
            priority_scores['hydration'] = 2
        
        # Pigmentation Analysis
        melanin_level = analysis_results.get('melaninLevel', 50)
        if melanin_level > 70:
            skin_concerns.extend(['pigmentation', 'dark spots', 'uneven tone'])
            priority_scores['pigmentation'] = 2
        
        # Always recommend sun protection
        skin_concerns.append('UV protection')
        if 'UV protection' not in priority_scores:
            priority_scores['UV protection'] = 2
        
        return skin_concerns, priority_scores
    
    def get_recommendations_by_skin_analysis(self, analysis_results):
        """Get product recommendations based on skin analysis"""
        skin_concerns, priority_scores = self.analyze_skin_needs(analysis_results)
        
        recommendations = []
        
        # Score products based on matching concerns
        for _, product in self.products_df.iterrows():
            score = 0
            matched_concerns = []
            product_concerns = product['skin_concerns'].lower()
            
            for concern in skin_concerns:
                if concern.lower() in product_concerns:
                    concern_priority = priority_scores.get(concern, 1)
                    score += concern_priority
                    matched_concerns.append(concern)
            
            # Bonus points for highly rated products
            rating_bonus = (product['rating'] - 4.0) * 0.5 if product['rating'] > 4.0 else 0
            score += rating_bonus
            
            if score > 0:
                product_dict = product.to_dict()
                product_dict['recommendation_score'] = score
                product_dict['matched_concerns'] = matched_concerns
                product_dict['priority_level'] = max([priority_scores.get(c, 1) for c in matched_concerns], default=1)
                recommendations.append(product_dict)
        
        # Sort by recommendation score and priority
        recommendations.sort(key=lambda x: (x['recommendation_score'], x['priority_level'], x['rating']), reverse=True)
        
        # Ensure variety in product categories
        final_recommendations = self.ensure_category_diversity(recommendations)
        
        return final_recommendations[:6]  # Return top 6 recommendations
    
    def ensure_category_diversity(self, recommendations):
        """Ensure we have diverse product categories in recommendations"""
        category_count = {}
        diverse_recommendations = []
        
        for product in recommendations:
            category = product['category']
            if category_count.get(category, 0) < 2:  # Max 2 products per category
                diverse_recommendations.append(product)
                category_count[category] = category_count.get(category, 0) + 1
        
        return diverse_recommendations
    
    def create_skincare_routine(self, analysis_results):
        """Create a complete skincare routine"""
        recommendations = self.get_recommendations_by_skin_analysis(analysis_results)
        
        routine = {
            'morning': [],
            'evening': [],
            'weekly': []
        }
        
        # Categorize products by usage time
        for product in recommendations:
            usage = product.get('usage', '').lower()
            
            if 'morning' in usage or product['category'] == 'Sunscreen':
                routine['morning'].append(product)
            elif 'evening' in usage or 'retinol' in product['name'].lower():
                routine['evening'].append(product)
            elif 'weekly' in usage or product['category'] == 'Exfoliant':
                routine['weekly'].append(product)
            else:
                # Default to both morning and evening
                routine['morning'].append(product)
                routine['evening'].append(product)
        
        return routine
    
    def get_product_explanation(self, product, analysis_results):
        """Generate explanation for why a product is recommended"""
        explanations = []
        
        # Check analysis results and match to product benefits
        if analysis_results.get('uvDamage', 0) > 50 and 'UV' in product.get('skin_concerns', ''):
            explanations.append("Your skin shows UV damage - this product helps repair and protect against further damage")
        
        if analysis_results.get('acneSeverity', 'Minimal') != 'Minimal' and 'acne' in product.get('skin_concerns', '').lower():
            explanations.append("This product targets acne and helps clear blemishes based on your skin analysis")
        
        if analysis_results.get('hydrationScore', 100) < 50 and 'hydration' in product.get('skin_concerns', '').lower():
            explanations.append("Your skin needs more moisture - this product provides deep hydration")
        
        if analysis_results.get('wrinkleScore', 0) > 40 and any(word in product.get('skin_concerns', '').lower() for word in ['aging', 'wrinkles', 'fine lines']):
            explanations.append("This anti-aging product helps reduce fine lines and wrinkles detected in your analysis")
        
        # Default explanation if no specific matches
        if not explanations:
            explanations.append("This highly-rated product addresses your skin type and general concerns")
        
        return explanations[0] if explanations else "Recommended based on your skin analysis"
    
    def generate_amazon_search_url(self, product):
        """Generate Amazon search URL with product-specific query (URL-encoded)."""
        base_url = "https://www.amazon.com/s"
        name = (product.get('name') or '').strip()
        brand = (product.get('brand') or '').strip()
        search_query = product.get('search_query') or f"{name} {brand}".strip() or ''
        params = {
            'k': search_query,
            'ref': 'sr_pg_1'
        }
        return f"{base_url}?{urlencode(params, doseq=True, quote_via=quote_plus)}"

    def get_alternative_purchase_links(self, product):
        """Return a dict of safe, URL-encoded alternative purchase/search links."""
        name = (product.get('name') or '').strip()
        brand = (product.get('brand') or '').strip()
        combined = f"{name} {brand}".strip() or product.get('search_query', '').strip() or ''
        q = quote_plus(combined) if combined else ''

        alternatives = {
            'amazon': self.generate_amazon_search_url(product) if combined else "https://www.amazon.com/",
            'walmart': f"https://www.walmart.com/search/?query={q}" if q else "https://www.walmart.com/",
            'target': f"https://www.target.com/s?searchTerm={q}" if q else "https://www.target.com/",
            'sephora': f"https://www.sephora.com/search?keyword={q}" if q else "https://www.sephora.com/"
        }
        return alternatives
