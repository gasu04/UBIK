"""
UBIK Ingestion System - Content Classifier

Classifies content chunks as EPISODIC (experiences) or SEMANTIC (beliefs/values).
Uses rule-based pattern matching with confidence scoring.

Classification Logic:
    EPISODIC: Letters, narratives, events, conversations, dated experiences
    SEMANTIC: Beliefs, values, principles, self-definitions, timeless truths

Features:
    - Memory type classification with confidence
    - Theme extraction (family, legacy, values, etc.)
    - Participant detection (spouse, children, etc.)
    - Emotional valence analysis
    - Importance scoring
    - Event/knowledge type detection

Usage:
    from ingest.classifiers import ContentClassifier
    from ingest.chunkers import Chunk

    classifier = ContentClassifier()
    candidate = classifier.classify(chunk, processed_content)
    print(f"{candidate.memory_type}: {candidate.confidence:.2f}")

Version: 0.1.0
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from .chunkers import Chunk
from .models import MemoryCandidate, MemoryType, ProcessedContent

__all__ = [
    'ContentClassifier',
    'ClassifierConfig',
]


# =============================================================================
# Pattern Definitions
# =============================================================================

# EPISODIC indicators - experiences, events, narratives
EPISODIC_PATTERNS = {
    # Letter patterns
    "letter_opening": re.compile(
        r'\b(Dear|Dearest|My dear|To my|'
        r'Querido|Querida|Mi querido|Mi querida|A mi)\b',
        re.IGNORECASE
    ),
    "letter_closing": re.compile(
        r'\b(Love,|With love,|Yours truly,|Sincerely,|Your loving|'
        r'All my love,|Forever yours,|With all my heart|'
        r'Con amor,|Con cariño,|Tuyo,|Tuya,|Con todo mi amor,|'
        r'Para siempre tuyo,|Para siempre tuya,|Con todo mi cariño)\b',
        re.IGNORECASE
    ),
    "salutation": re.compile(
        r'\b(Dear\s+\w+|Hi\s+\w+|Hello\s+\w+|'
        r'Querido\s+\w+|Querida\s+\w+|Hola\s+\w+)\b',
        re.IGNORECASE
    ),

    # Date/time references
    "date_reference": re.compile(
        r'\b(Today|Yesterday|Tomorrow|Last (week|month|year)|'
        r'This (morning|afternoon|evening)|On (Monday|Tuesday|Wednesday|'
        r'Thursday|Friday|Saturday|Sunday)|In (January|February|March|'
        r'April|May|June|July|August|September|October|November|December)|'
        r'Hoy|Ayer|Mañana|La semana pasada|El mes pasado|El año pasado|'
        r'Esta (mañana|tarde|noche)|El (lunes|martes|miércoles|jueves|'
        r'viernes|sábado|domingo)|En (enero|febrero|marzo|abril|mayo|'
        r'junio|julio|agosto|septiembre|octubre|noviembre|diciembre)|'
        r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4})\b',
        re.IGNORECASE
    ),
    "temporal_marker": re.compile(
        r'\b(When I was|Back in|During|At the time|That day|'
        r'That night|One day|Once upon|I remember when|'
        r'Cuando era|De vuelta en|Durante|En ese momento|Ese día|'
        r'Esa noche|Un día|Una vez|Recuerdo cuando|Hace años|Hace tiempo)\b',
        re.IGNORECASE
    ),

    # Past tense narrative
    "past_narrative": re.compile(
        r'\b(I (went|saw|heard|felt|met|visited|attended|experienced|'
        r'discovered|learned|realized|decided|chose|made|took|gave|'
        r'told|asked|said|thought|knew|understood|remembered)|'
        r'(fui|vi|escuché|sentí|conocí|visité|asistí|experimenté|'
        r'descubrí|aprendí|me di cuenta|decidí|elegí|hice|tomé|di|'
        r'dije|pregunté|pensé|supe|entendí|recordé))\b',
        re.IGNORECASE
    ),

    # Dialogue markers
    "dialogue": re.compile(
        r'("[^"]+"|\'[^\']+\'|\bsaid\b|\basked\b|\breplied\b|\btold me\b|'
        r'\bdijo\b|\bpreguntó\b|\brespondió\b|\bme dijo\b)',
        re.IGNORECASE
    ),

    # Therapy/session patterns
    "therapy": re.compile(
        r'\b(therapy|session|therapist|counselor|talked about|'
        r'we discussed|worked through|breakthrough|'
        r'terapia|sesión|terapeuta|consejero|consejera|hablamos de|'
        r'discutimos|trabajamos|avance|avanzamos)\b',
        re.IGNORECASE
    ),

    # Event descriptions
    "event": re.compile(
        r'\b(wedding|funeral|birthday|anniversary|graduation|'
        r'ceremony|celebration|party|reunion|vacation|trip|'
        r'hospital|surgery|diagnosis|meeting|interview|'
        r'boda|funeral|cumpleaños|aniversario|graduación|'
        r'ceremonia|celebración|fiesta|reunión|vacaciones|viaje|'
        r'hospital|cirugía|diagnóstico|cita|entrevista)\b',
        re.IGNORECASE
    ),

    # Specific memories
    "memory_marker": re.compile(
        r'\b(I remember|I recall|I\'ll never forget|'
        r'It was the first time|That was when|The moment|'
        r'Recuerdo|Me acuerdo|Nunca olvidaré|'
        r'Fue la primera vez|Fue cuando|El momento en que)\b',
        re.IGNORECASE
    ),
}

# SEMANTIC indicators - beliefs, values, principles
SEMANTIC_PATTERNS = {
    # Belief statements
    "belief": re.compile(
        r'\b(I believe|I think|My view is|In my opinion|'
        r'I\'m convinced|I hold that|My belief is|'
        r'Creo|Creo que|Pienso|Mi opinión es|En mi opinión|'
        r'Estoy convencido|Estoy convencida|Para mí)\b',
        re.IGNORECASE
    ),

    # Value statements
    "value": re.compile(
        r'\b(I value|My (core )?values?|What matters most|'
        r'Most important to me|I prioritize|I treasure|'
        r'Valoro|Mis valores|Lo que más importa|'
        r'Lo más importante para mí|Priorizo|Aprecio)\b',
        re.IGNORECASE
    ),

    # Preference patterns
    "preference": re.compile(
        r'\b(I prefer|I always|I never|I tend to|'
        r'My preference|I choose to|I make it a point|'
        r'Prefiero|Siempre|Nunca|Tiendo a|Mi preferencia|Elijo)\b',
        re.IGNORECASE
    ),

    # Philosophical statements
    "philosophy": re.compile(
        r'\b(The meaning of|The purpose of|Life is|'
        r'Truth is|Reality is|What defines|The essence of|'
        r'El significado de|El propósito de|La vida es|'
        r'La verdad es|La realidad es|Lo que define|La esencia de)\b',
        re.IGNORECASE
    ),

    # Self-definition
    "self_definition": re.compile(
        r'\b(I am (a |the kind of )?person who|My identity|'
        r'At my core|Who I am|What makes me|'
        r'Soy (el tipo de )?persona que|Mi identidad|'
        r'En mi núcleo|Quien soy|Lo que me hace)\b',
        re.IGNORECASE
    ),

    # Timeless principles
    "principle": re.compile(
        r'\b(One should|It\'s important to|The key is|'
        r'The secret to|A person should|We must|'
        r'It matters that|What counts is|'
        r'Se debe|Es importante|La clave es|'
        r'El secreto para|Una persona debe|Debemos|'
        r'Lo que importa es|Lo que cuenta es)\b',
        re.IGNORECASE
    ),

    # Character/trait descriptions
    "character": re.compile(
        r'\b(I\'m (generally |usually |naturally )?(patient|kind|'
        r'honest|loyal|determined|curious|creative|compassionate|'
        r'resilient|optimistic|practical)|'
        r'Soy (generalmente |usualmente |naturalmente )?(paciente|amable|'
        r'honesto|honesta|leal|determinado|determinada|curioso|curiosa|'
        r'creativo|creativa|compasivo|compasiva|resiliente|optimista|práctico|práctica))\b',
        re.IGNORECASE
    ),

    # Lessons learned (timeless)
    "lesson": re.compile(
        r'\b(I\'ve learned that|Life has taught me|'
        r'Experience shows|The lesson is|I\'ve come to understand|'
        r'He aprendido que|La vida me ha enseñado|'
        r'La experiencia muestra|La lección es|He llegado a entender|'
        r'Aprendí que|Con el tiempo aprendí)\b',
        re.IGNORECASE
    ),
}

# Theme keywords
THEME_KEYWORDS: Dict[str, List[str]] = {
    "family": [
        # English
        "family", "families", "children", "child", "son", "daughter",
        "grandchildren", "grandchild", "grandson", "granddaughter",
        "spouse", "wife", "husband", "parent", "parents", "mother",
        "father", "mom", "dad", "sibling", "brother", "sister",
        "aunt", "uncle", "cousin", "relative", "ancestor",
        # Spanish
        "familia", "familias", "hijos", "hijo", "hija", "niños", "niño", "niña",
        "esposo", "esposa", "pareja", "madre", "padre", "mamá", "papá",
        "hermano", "hermana", "abuelo", "abuela", "nieto", "nieta",
        "tío", "tía", "primo", "prima", "pariente", "ancestro",
    ],
    "legacy": [
        # English
        "legacy", "remember", "remembered", "future", "generations",
        "pass on", "leave behind", "heritage", "tradition", "traditions",
        "inherit", "inheritance", "lasting", "enduring", "continue",
        # Spanish
        "legado", "recuerdo", "recordar", "futuro", "generaciones",
        "pasar", "dejar", "herencia", "tradición", "tradiciones",
        "heredar", "duradero", "duradera", "continuar",
    ],
    "authenticity": [
        # English
        "authentic", "genuine", "true self", "honest", "honesty",
        "real", "sincere", "integrity", "transparent", "vulnerable",
        "truth", "truthful", "pretend", "mask", "facade",
        # Spanish
        "auténtico", "auténtica", "genuino", "genuina", "verdadero", "verdadera",
        "honesto", "honesta", "honestidad", "real", "sincero", "sincera",
        "integridad", "transparente", "vulnerable", "verdad", "máscara",
    ],
    "values": [
        # English
        "value", "values", "believe", "belief", "beliefs", "principle",
        "principles", "ethics", "ethical", "moral", "morals",
        "right", "wrong", "virtue", "virtues", "character",
        # Spanish
        "valor", "valores", "creer", "creencia", "creencias", "principio",
        "principios", "ética", "ético", "ética", "moral", "morales",
        "correcto", "incorrecto", "virtud", "virtudes", "carácter",
    ],
    "love": [
        # English
        "love", "loved", "loving", "care", "caring", "affection",
        "cherish", "cherished", "adore", "devoted", "devotion",
        "heart", "heartfelt", "tender", "warmth", "embrace",
        # Spanish
        "amor", "amado", "amada", "amar", "amando", "cariño", "cariñoso",
        "cariñosa", "afecto", "adorar", "devoto", "devota", "devoción",
        "corazón", "ternura", "calidez", "abrazo",
    ],
    "philosophy": [
        # English
        "meaning", "purpose", "identity", "consciousness", "existence",
        "life", "death", "wisdom", "understanding", "truth",
        "reality", "spiritual", "soul", "essence", "nature",
        # Spanish
        "significado", "propósito", "identidad", "conciencia", "existencia",
        "vida", "muerte", "sabiduría", "comprensión", "verdad",
        "realidad", "espiritual", "alma", "esencia", "naturaleza",
    ],
    "health": [
        # English
        "health", "healthy", "illness", "sick", "disease", "cancer",
        "diagnosis", "treatment", "doctor", "hospital", "medical",
        "wellness", "healing", "recovery", "pain", "suffering",
        # Spanish
        "salud", "saludable", "enfermedad", "enfermo", "enferma", "cáncer",
        "diagnóstico", "tratamiento", "médico", "médica", "hospital",
        "bienestar", "curación", "recuperación", "dolor", "sufrimiento",
    ],
    "career": [
        # English
        "career", "work", "job", "profession", "business", "company",
        "office", "colleague", "boss", "employee", "retire",
        "retirement", "success", "achievement", "accomplishment",
        # Spanish
        "carrera", "trabajo", "empleo", "profesión", "negocio", "empresa",
        "oficina", "colega", "jefe", "jefa", "empleado", "empleada",
        "jubilación", "jubilarse", "éxito", "logro", "meta",
    ],
    "growth": [
        # English
        "growth", "grow", "growing", "learn", "learning", "change",
        "changing", "evolve", "evolving", "develop", "development",
        "improve", "improvement", "progress", "journey", "path",
        # Spanish
        "crecimiento", "crecer", "creciendo", "aprender", "aprendizaje",
        "cambio", "cambiar", "evolucionar", "evolucionando", "desarrollar",
        "desarrollo", "mejorar", "mejora", "progreso", "camino", "trayectoria",
    ],
    "spirituality": [
        # English
        "god", "faith", "prayer", "church", "spiritual", "spirit",
        "soul", "blessing", "blessed", "divine", "sacred", "holy",
        "worship", "religion", "religious", "believe",
        # Spanish
        "dios", "fe", "oración", "iglesia", "espiritual", "espíritu",
        "alma", "bendición", "bendito", "bendita", "divino", "divina",
        "sagrado", "sagrada", "adoración", "religión", "religioso", "religiosa",
    ],
    "therapy": [
        # English
        "therapy", "session", "therapist", "counseling", "anxiety",
        "depression", "trauma", "healing", "mental health",
        # Spanish
        "terapia", "sesión", "terapeuta", "consejería", "ansiedad",
        "depresión", "trauma", "sanación", "salud mental",
    ],
    "relationships": [
        # English
        "relationship", "relationships", "friend", "friends", "partner",
        "connection", "trust", "bond", "communication",
        # Spanish
        "relación", "relaciones", "amigo", "amiga", "amigos", "amigas",
        "pareja", "conexión", "confianza", "vínculo", "comunicación",
    ],
}

# Participant keywords
PARTICIPANT_KEYWORDS: Dict[str, List[str]] = {
    "spouse": [
        "spouse", "wife", "husband", "partner",
        "esposo", "esposa", "pareja", "cónyuge",
    ],
    "children": [
        "children", "child", "son", "daughter", "kids", "kid",
        "hijos", "hijo", "hija", "niños", "niño", "niña", "chicos",
    ],
    "grandchildren": [
        "grandchildren", "grandchild", "grandson", "granddaughter",
        "grandkids", "grandkid",
        "nietos", "nieto", "nieta",
    ],
    "parents": [
        "mother", "father", "mom", "dad", "parent", "parents",
        "madre", "padre", "mamá", "papá", "padres",
    ],
    "siblings": [
        "brother", "sister", "sibling", "siblings",
        "hermano", "hermana", "hermanos",
    ],
    "therapist": [
        "therapist", "counselor", "psychologist", "psychiatrist",
        "terapeuta", "consejero", "consejera", "psicólogo", "psicóloga", "psiquiatra",
    ],
    "doctor": [
        "doctor", "physician", "nurse", "oncologist", "specialist",
        "médico", "médica", "enfermero", "enfermera", "oncólogo", "especialista",
    ],
    "friends": [
        "friend", "friends", "buddy", "pal",
        "amigo", "amiga", "amigos", "amigas",
    ],
    "family": [
        "family", "relative", "relatives", "aunt", "uncle", "cousin",
        "familia", "familiar", "familiares", "tío", "tía", "primo", "prima",
    ],
}

# Emotional valence keywords
EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "positive": [
        # English
        "happy", "happiness", "joy", "joyful", "love", "loving",
        "grateful", "gratitude", "thankful", "excited", "excitement",
        "proud", "pride", "blessed", "wonderful", "amazing", "beautiful",
        "hopeful", "hope", "peaceful", "peace", "content", "satisfied",
        "delighted", "thrilled", "elated", "cheerful", "optimistic",
        # Spanish
        "feliz", "felicidad", "alegría", "alegre", "amor", "amando",
        "agradecido", "agradecida", "gratitud", "gracias", "emocionado",
        "emocionada", "emoción", "orgulloso", "orgullosa", "orgullo",
        "bendecido", "bendecida", "maravilloso", "maravillosa", "asombroso",
        "hermoso", "hermosa", "esperanzador", "esperanzadora", "esperanza",
        "tranquilo", "tranquila", "paz", "contento", "contenta",
        "satisfecho", "satisfecha", "encantado", "encantada", "optimista",
    ],
    "negative": [
        # English
        "sad", "sadness", "angry", "anger", "frustrated", "frustration",
        "worried", "worry", "anxious", "anxiety", "disappointed",
        "disappointment", "hurt", "pain", "painful", "grief", "grieving",
        "scared", "fear", "fearful", "depressed", "depression", "lonely",
        "loneliness", "regret", "guilty", "shame", "ashamed",
        # Spanish
        "triste", "tristeza", "enojado", "enojada", "enojo", "rabia",
        "frustrado", "frustrada", "frustración", "preocupado", "preocupada",
        "preocupación", "ansioso", "ansiosa", "ansiedad", "decepcionado",
        "decepcionada", "decepción", "herido", "herida", "dolor", "doloroso",
        "duelo", "llorando", "asustado", "asustada", "miedo", "temeroso",
        "temerosa", "deprimido", "deprimida", "depresión", "solo", "sola",
        "soledad", "arrepentimiento", "culpa", "vergüenza", "avergonzado",
        "avergonzada", "angustiado", "angustiada",
    ],
    "reflective": [
        # English
        "think", "thinking", "thought", "wonder", "wondering",
        "consider", "considering", "realize", "realized", "understand",
        "understanding", "ponder", "reflect", "reflecting", "contemplate",
        "muse", "question", "questioning",
        # Spanish
        "pensar", "pensando", "pensamiento", "preguntarme", "preguntarse",
        "considerar", "considerando", "darme cuenta", "comprender",
        "comprensión", "reflexionar", "reflexionando", "contemplar",
        "meditar", "cuestionar", "cuestionando", "analizar",
    ],
}

# Event types for episodic memories
EVENT_TYPES: Dict[str, List[str]] = {
    "milestone": [
        "wedding", "birth", "graduation", "retirement", "anniversary",
        "birthday", "promotion", "achievement",
        "boda", "nacimiento", "graduación", "jubilación", "aniversario",
        "cumpleaños", "ascenso", "logro", "meta alcanzada",
    ],
    "loss": [
        "death", "funeral", "dying", "passed away", "lost", "goodbye",
        "mourning", "grief",
        "muerte", "funeral", "muriendo", "falleció", "perdido", "adiós",
        "duelo", "luto",
    ],
    "medical": [
        "diagnosis", "surgery", "hospital", "treatment", "cancer",
        "illness", "recovery", "therapy",
        "diagnóstico", "cirugía", "hospital", "tratamiento", "cáncer",
        "enfermedad", "recuperación", "terapia",
    ],
    "travel": [
        "trip", "vacation", "travel", "journey", "visit", "visited",
        "flew", "drove", "destination",
        "viaje", "vacaciones", "viajar", "visita", "visitar",
        "voló", "manejó", "destino",
    ],
    "conversation": [
        "talked", "discussed", "conversation", "said", "told",
        "asked", "replied", "shared",
        "hablamos", "discutimos", "conversación", "dijo", "dije",
        "contó", "preguntó", "respondió", "compartió",
    ],
    "activity": [
        "played", "cooked", "built", "made", "created", "worked",
        "helped", "taught", "learned",
        "jugó", "cocinó", "construyó", "hizo", "creó", "trabajó",
        "ayudó", "enseñó", "aprendió",
    ],
    "gathering": [
        "party", "reunion", "celebration", "ceremony", "meeting",
        "dinner", "gathering", "get-together",
        "fiesta", "reunión", "celebración", "ceremonia", "cita",
        "cena", "evento", "encuentro",
    ],
}

# Knowledge types for semantic memories
KNOWLEDGE_TYPES: Dict[str, List[str]] = {
    "belief": [
        "believe", "belief", "faith", "conviction", "trust",
        "confident", "certain",
        "creer", "creencia", "fe", "convicción", "confiar", "seguro", "cierto",
    ],
    "value": [
        "value", "values", "important", "matters", "priority",
        "principle", "cherish",
        "valor", "valores", "importante", "importa", "prioridad",
        "principio", "apreciar",
    ],
    "insight": [
        "realize", "understand", "learn", "discover", "recognize",
        "see", "know",
        "darme cuenta", "comprender", "aprender", "descubrir",
        "reconocer", "ver", "saber",
    ],
    "preference": [
        "prefer", "like", "enjoy", "love", "favor", "choose",
        "tendency",
        "preferir", "gustar", "disfrutar", "amar", "favorecer",
        "elegir", "tendencia",
    ],
    "identity": [
        "am", "identity", "self", "person", "character", "nature",
        "essence",
        "soy", "identidad", "yo mismo", "yo misma", "persona",
        "carácter", "naturaleza", "esencia",
    ],
    "wisdom": [
        "wisdom", "lesson", "truth", "meaning", "purpose",
        "philosophy", "principle",
        "sabiduría", "lección", "verdad", "significado", "propósito",
        "filosofía", "principio",
    ],
}


@dataclass
class ClassifierConfig:
    """
    Configuration for content classification.

    Attributes:
        episodic_threshold: Minimum score to classify as episodic
        semantic_threshold: Minimum score to classify as semantic
        min_confidence: Minimum confidence to avoid SKIP
        max_themes: Maximum themes to extract per chunk
        importance_boost_themes: Themes that boost importance
    """
    episodic_threshold: float = 0.4
    semantic_threshold: float = 0.4
    min_confidence: float = 0.3
    max_themes: int = 5
    importance_boost_themes: List[str] = field(
        default_factory=lambda: ["family", "legacy", "values", "love"]
    )


class ContentClassifier:
    """
    Classifies content chunks as EPISODIC or SEMANTIC memories.

    Uses pattern matching and keyword analysis to determine:
    - Memory type (episodic vs semantic vs skip)
    - Confidence score
    - Themes and topics
    - Participants mentioned
    - Emotional valence
    - Importance score

    Attributes:
        config: Classification configuration

    Example:
        classifier = ContentClassifier()
        candidate = classifier.classify(chunk, processed_content)

        if candidate.memory_type == MemoryType.EPISODIC:
            print(f"Event type: {candidate.event_type}")
            print(f"Participants: {candidate.participants}")
        else:
            print(f"Knowledge type: {candidate.knowledge_type}")
    """

    def __init__(self, config: Optional[ClassifierConfig] = None):
        """Initialize classifier with configuration."""
        self.config = config or ClassifierConfig()

        # Pre-compile word sets for faster lookup
        self._theme_words: Dict[str, Set[str]] = {
            theme: set(words)
            for theme, words in THEME_KEYWORDS.items()
        }
        self._participant_words: Dict[str, Set[str]] = {
            role: set(words)
            for role, words in PARTICIPANT_KEYWORDS.items()
        }
        self._emotion_words: Dict[str, Set[str]] = {
            valence: set(words)
            for valence, words in EMOTION_KEYWORDS.items()
        }
        self._event_words: Dict[str, Set[str]] = {
            etype: set(words)
            for etype, words in EVENT_TYPES.items()
        }
        self._knowledge_words: Dict[str, Set[str]] = {
            ktype: set(words)
            for ktype, words in KNOWLEDGE_TYPES.items()
        }

    def classify(
        self,
        chunk: Chunk,
        context: ProcessedContent
    ) -> MemoryCandidate:
        """
        Classify a chunk and create a MemoryCandidate.

        Args:
            chunk: The content chunk to classify
            context: Full processed content for additional context

        Returns:
            MemoryCandidate with classification results
        """
        text = chunk.text
        text_lower = text.lower()
        words = set(text_lower.split())

        # Detect memory type and confidence
        memory_type, confidence = self._detect_memory_type(text, text_lower)

        # Extract metadata
        themes = self._extract_themes(words)
        participants = self._detect_participants(words)
        emotional_valence = self._detect_emotional_valence(words)
        importance = self._calculate_importance(
            themes, emotional_valence, confidence, chunk.word_count
        )

        # Type-specific detection
        event_type = None
        knowledge_type = None

        if memory_type == MemoryType.EPISODIC:
            event_type = self._detect_event_type(text_lower, words)
        elif memory_type == MemoryType.SEMANTIC:
            knowledge_type = self._detect_knowledge_type(text_lower, words)

        # Detect timestamp for episodic content
        timestamp = self._extract_timestamp(text) if memory_type == MemoryType.EPISODIC else None

        # Determine stability
        stability = self._determine_stability(memory_type, confidence)

        # Determine category from primary theme
        category = themes[0] if themes else "general"

        return MemoryCandidate(
            content=text,
            memory_type=memory_type,
            confidence=confidence,
            category=category,
            themes=themes,
            event_type=event_type,
            participants=participants,
            emotional_valence=emotional_valence,
            knowledge_type=knowledge_type,
            stability=stability,
            source_file=context.source_item.original_filename,
            source_chunk_index=chunk.index,
            timestamp=timestamp,
            importance=importance,
        )

    def _detect_memory_type(
        self,
        text: str,
        text_lower: str
    ) -> Tuple[MemoryType, float]:
        """
        Detect whether content is episodic or semantic.

        Returns tuple of (MemoryType, confidence).
        """
        episodic_score = 0.0
        semantic_score = 0.0
        episodic_matches = 0
        semantic_matches = 0

        # Check episodic patterns
        for name, pattern in EPISODIC_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                episodic_matches += len(matches)
                # Weight different patterns
                if name in ["letter_opening", "letter_closing"]:
                    episodic_score += 0.3 * len(matches)
                elif name in ["memory_marker", "temporal_marker"]:
                    episodic_score += 0.25 * len(matches)
                elif name in ["past_narrative"]:
                    episodic_score += 0.15 * len(matches)
                elif name in ["dialogue", "event"]:
                    episodic_score += 0.2 * len(matches)
                else:
                    episodic_score += 0.1 * len(matches)

        # Check semantic patterns
        for name, pattern in SEMANTIC_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                semantic_matches += len(matches)
                # Weight different patterns
                if name in ["belief", "value"]:
                    semantic_score += 0.3 * len(matches)
                elif name in ["self_definition", "principle"]:
                    semantic_score += 0.25 * len(matches)
                elif name in ["philosophy", "lesson"]:
                    semantic_score += 0.2 * len(matches)
                else:
                    semantic_score += 0.15 * len(matches)

        # Normalize scores
        max_score = max(episodic_score, semantic_score, 0.1)
        episodic_score = min(episodic_score / max_score, 1.0) if episodic_score > 0 else 0
        semantic_score = min(semantic_score / max_score, 1.0) if semantic_score > 0 else 0

        # Determine type and confidence
        if episodic_score > semantic_score and episodic_score >= self.config.episodic_threshold:
            confidence = episodic_score * 0.7 + 0.3  # Scale to 0.3-1.0
            return MemoryType.EPISODIC, min(confidence, 1.0)

        elif semantic_score > episodic_score and semantic_score >= self.config.semantic_threshold:
            confidence = semantic_score * 0.7 + 0.3
            return MemoryType.SEMANTIC, min(confidence, 1.0)

        elif episodic_score > 0 or semantic_score > 0:
            # Low confidence but some signal
            if episodic_score >= semantic_score:
                confidence = max(episodic_score * 0.5 + 0.2, self.config.min_confidence)
                return MemoryType.EPISODIC, confidence
            else:
                confidence = max(semantic_score * 0.5 + 0.2, self.config.min_confidence)
                return MemoryType.SEMANTIC, confidence

        else:
            # No clear signal - default to semantic with low confidence
            # (Most general content tends to be factual/semantic)
            return MemoryType.SEMANTIC, self.config.min_confidence

    def _extract_themes(self, words: Set[str]) -> List[str]:
        """
        Extract themes from content based on keyword presence.

        Returns list of themes ordered by relevance.
        """
        theme_scores: Dict[str, int] = {}

        for theme, theme_words in self._theme_words.items():
            matches = words & theme_words
            if matches:
                theme_scores[theme] = len(matches)

        # Sort by score descending
        sorted_themes = sorted(
            theme_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [theme for theme, _ in sorted_themes[:self.config.max_themes]]

    def _detect_participants(self, words: Set[str]) -> List[str]:
        """
        Detect participants mentioned in content.

        Returns list of participant roles found.
        """
        participants = []

        for role, role_words in self._participant_words.items():
            if words & role_words:
                participants.append(role)

        return participants

    def _detect_emotional_valence(self, words: Set[str]) -> str:
        """
        Detect overall emotional valence of content.

        Returns: "positive", "negative", "reflective", "mixed", or "neutral"
        """
        scores = {
            valence: len(words & valence_words)
            for valence, valence_words in self._emotion_words.items()
        }

        total = sum(scores.values())
        if total == 0:
            return "neutral"

        # Check for mixed emotions
        non_zero = sum(1 for s in scores.values() if s > 0)
        if non_zero >= 2:
            max_score = max(scores.values())
            second_max = sorted(scores.values(), reverse=True)[1]
            if second_max >= max_score * 0.5:
                return "mixed"

        # Return dominant valence
        return max(scores.items(), key=lambda x: x[1])[0]

    def _calculate_importance(
        self,
        themes: List[str],
        emotional_valence: str,
        confidence: float,
        word_count: int
    ) -> float:
        """
        Calculate importance score for the content.

        Factors:
        - Theme importance (family, legacy boost)
        - Emotional intensity
        - Classification confidence
        - Content length
        """
        importance = 0.5  # Base importance

        # Theme boost
        for theme in themes:
            if theme in self.config.importance_boost_themes:
                importance += 0.1

        # Emotional intensity boost
        if emotional_valence in ["positive", "negative"]:
            importance += 0.1
        elif emotional_valence == "mixed":
            importance += 0.15  # Complex emotions often important

        # Confidence contribution
        importance += confidence * 0.1

        # Length factor (longer content often more substantive)
        if word_count > 100:
            importance += 0.05
        if word_count > 200:
            importance += 0.05

        return min(importance, 1.0)

    def _detect_event_type(
        self,
        text_lower: str,
        words: Set[str]
    ) -> Optional[str]:
        """
        Detect the type of event for episodic memories.

        Returns event type or None if not detected.
        """
        event_scores: Dict[str, int] = {}

        for event_type, event_words in self._event_words.items():
            matches = words & event_words
            if matches:
                event_scores[event_type] = len(matches)

        if not event_scores:
            return None

        return max(event_scores.items(), key=lambda x: x[1])[0]

    def _detect_knowledge_type(
        self,
        text_lower: str,
        words: Set[str]
    ) -> Optional[str]:
        """
        Detect the type of knowledge for semantic memories.

        Returns knowledge type or None if not detected.
        """
        knowledge_scores: Dict[str, int] = {}

        for knowledge_type, knowledge_words in self._knowledge_words.items():
            matches = words & knowledge_words
            if matches:
                knowledge_scores[knowledge_type] = len(matches)

        if not knowledge_scores:
            return None

        return max(knowledge_scores.items(), key=lambda x: x[1])[0]

    def _extract_timestamp(self, text: str) -> Optional[datetime]:
        """
        Attempt to extract a timestamp from the text.

        Returns datetime if found, None otherwise.
        """
        # Look for explicit dates
        date_patterns = [
            # MM/DD/YYYY or MM-DD-YYYY
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', lambda m: (
                int(m.group(3)), int(m.group(1)), int(m.group(2))
            )),
            # YYYY-MM-DD
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: (
                int(m.group(1)), int(m.group(2)), int(m.group(3))
            )),
            # Month DD, YYYY
            (r'(January|February|March|April|May|June|July|August|'
             r'September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
             lambda m: self._parse_month_date(m)),
        ]

        for pattern, extractor in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    year, month, day = extractor(match)
                    return datetime(year, month, day)
                except (ValueError, TypeError):
                    continue

        return None

    def _parse_month_date(self, match) -> Tuple[int, int, int]:
        """Parse a month name date match."""
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        month = months[match.group(1).lower()]
        day = int(match.group(2))
        year = int(match.group(3))
        return (year, month, day)

    def _determine_stability(
        self,
        memory_type: MemoryType,
        confidence: float
    ) -> str:
        """
        Determine memory stability classification.

        Returns: "stable", "evolving", or "uncertain"
        """
        if memory_type == MemoryType.SEMANTIC:
            # Semantic memories are generally more stable
            if confidence >= 0.7:
                return "stable"
            elif confidence >= 0.5:
                return "evolving"
            else:
                return "uncertain"
        else:
            # Episodic memories
            if confidence >= 0.8:
                return "stable"
            elif confidence >= 0.5:
                return "evolving"
            else:
                return "uncertain"
