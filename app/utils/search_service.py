def search_for_multiple_terms(terms: str) -> list[str]:
    """
    Process search terms and return a list of cleaned search terms.
    
    Args:
        terms (str): String of terms to search for
        
    Returns:
        List[str]: List of cleaned search terms
    """
    if not terms:
        return []
    
    # Split by spaces and filter out empty strings
    terms_list = terms.split()
    
    # Remove any extra whitespace
    cleaned_terms = [term.strip() for term in terms_list if term.strip()]
    
    return cleaned_terms