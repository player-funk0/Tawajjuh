"""
dataset.py -- MASSIVELY EXPANDED version with 2500+ examples.
Uses data augmentation: paraphrasing, filler words, typos, abbreviations,
reordering, and code-switching variations to simulate real speech patterns.
"""

import re
import json
import hashlib
import numpy as np

from lexicon import APP_LEXICON, FILE_LEXICON, is_arabic

# =============================================================================
# EXPANDED ENGLISH APP TEMPLATES (60+)
# =============================================================================
APP_TEMPLATES_EN = [
    # Basic
    "open {x}", "launch {x}", "start {x}", "run {x}",
    "please open {x}", "can you open {x}", "could you open {x}",
    "would you mind opening {x}", "would you open {x} for me",
    "I want to open {x}", "I need {x} open", "I need to open {x}",
    "let me use {x}", "let me open {x}", "get {x} running",
    "fire up {x}", "boot up {x}", "load {x}", "bring up {x}",
    "show me {x}", "start up {x}", "go ahead and start {x}",
    # Questions
    "can you launch {x}", "could you launch {x}", "can you start {x}",
    "can we open {x}", "is it possible to open {x}",
    # Casual/urgent
    "open {x} now", "open {x} please", "open {x} quickly",
    "quick open {x}", "open {x} asap", "need {x} now",
    "{x} now", "{x} please", "{x} quickly",
    # With filler words
    "hey open {x}", "yo open {x}", "okay open {x}",
    "alright open {x}", "so open {x}", "just open {x}",
    # With articles
    "open the {x}", "open that {x}", "open this {x}",
    "launch the {x} app", "start the {x} application",
    # Indirect
    "I want {x} opened", "I would like to open {x}",
    "make {x} run", "get {x} to work", "turn on {x}",
    # Slang/abbreviated
    "open {x} app", "run {x} app", "start {x} program",
    "gimme {x}", "pass me {x}", "hand me {x}",
]

# =============================================================================
# EXPANDED ENGLISH FILE TEMPLATES (60+)
# =============================================================================
FILE_TEMPLATES_EN = [
    # Basic
    "open {x}", "open the file {x}", "open file called {x}",
    "open my file {x}", "open document {x}", "open the document {x}",
    "please open {x}", "can you open {x}", "could you open {x}",
    "can you open the file {x}", "can you open my {x}",
    "show me {x}", "show me the file {x}", "show me my {x}",
    "I need to see {x}", "I want to see {x}", "let me see {x}",
    "display {x}", "view {x}", "access {x}", "retrieve {x}",
    # Questions
    "where is {x}", "where is my {x}", "where did I save {x}",
    "can you find {x}", "can you show me {x}", "do you see {x}",
    # With location
    "open {x} from desktop", "open {x} from downloads",
    "open {x} from documents", "open {x} from folder",
    # Casual/urgent
    "open {x} now", "open {x} please", "open {x} quickly",
    "pull up {x}", "pull up the {x}", "find and open {x}",
    "get me {x}", "bring me {x}", "hand me {x}",
    # With articles/determiners
    "open that {x}", "open this {x}", "open the {x} file",
    "show that {x}", "show this {x}", "show the {x}",
    # Indirect
    "I am looking for {x}", "I need {x}", "I want {x}",
    "where did I put {x}", "I saved {x} somewhere",
    # Slang
    "wheres {x}", "find {x}", "locate {x}", "get {x}",
    "{x} file open it", "{x} open it", "{x} show me",
    # With adjectives
    "open the latest {x}", "open the new {x}", "open my recent {x}",
]

# =============================================================================
# EXPANDED ARABIC APP TEMPLATES (50+)
# =============================================================================
APP_TEMPLATES_AR = [
    # Basic
    "افتح {x}", "شغل {x}", "ابدأ {x}", "افتحلي {x}", "شغللي {x}",
    "ممكن تفتح {x}", "ممكن تشغل {x}", "لو سمحت افتح {x}",
    "عايز افتح {x}", "عايز اشغل {x}", "عايز {x} يشتغل",
    "افتح لي {x}", "شغل لي {x}", "ابدأ لي {x}",
    # Questions
    "ممكن تفتحلي {x}", "ممكن تشغللي {x}", "تقدر تفتح {x}",
    "تعرف تفتح {x}", "تقدر تشغل {x}",
    # Casual/urgent
    "افتح {x} دلوقتي", "افتح {x} بسرعة", "شغل {x} بسرعة",
    "يلا افتح {x}", "افتح {x} على طول", "شغل {x} على طول",
    # With filler
    "يا عم افتح {x}", "يا سيدي افتح {x}", "طيب افتح {x}",
    "خلاص افتح {x}", "اعمل افتح {x}", "اعمل شغل {x}",
    # Indirect
    "عايز استخدم {x}", "محتاج افتح {x}", "محتاج {x} يشتغل",
    "مش عارف افتح {x}", "مش عارف اشغل {x}",
    # With particles
    "افتح {x} بقى", "شغل {x} بقى", "افتح {x} كده",
    "افتح {x} من فضلك", "شغل {x} من فضلك",
    # Colloquial variations
    "افتح {x} عشاني", "شغل {x} عشاني", "{x} شغله",
    "{x} افتحه", "{x} شغلهولي", "اعمل run لـ {x}",
    # Definite
    "افتح ال{x}", "شغل ال{x}", "افتحلي ال{x}", "شغللي ال{x}",
]

# =============================================================================
# EXPANDED ARABIC FILE TEMPLATES (50+)
# =============================================================================
FILE_TEMPLATES_AR = [
    # Basic
    "افتح {x}", "افتح ملف {x}", "افتح الملف {x}",
    "ورّيني {x}", "وريني {x}", "وريني ملف {x}",
    "عايز اشوف {x}", "عايز اشوف ملف {x}", "عايز افتح {x}",
    "ممكن تفتح ملف {x}", "ممكن توريني {x}", "ممكن تشوفلي {x}",
    "هات {x}", "هاتلي {x}", "جيب {x}", "جيبلي {x}",
    # Questions
    "فين {x}", "فين ملف {x}", "فين {x} اللي عندك",
    "شوفت {x}", "شوفت ملف {x}", "لقيت {x}",
    # With location
    "افتح {x} اللي على الديسكتوب", "افتح {x} اللي في الدونلودز",
    "افتح {x} اللي في الملفات", "افتح {x} اللي حفظته",
    # Casual
    "وريني {x} بسرعة", "افتح {x} بسرعة", "هات {x} بسرعة",
    "يا عم وريني {x}", "يا سيدي افتح {x}", "طيب وريني {x}",
    # Indirect
    "عايز اشوف {x} ده", "عايز افتح {x} ده",
    "محتاج اشوف {x}", "محتاج افتح {x}",
    "مش لاقي {x}", "مش لاقي ملف {x}", "مش عارف لاقي {x}",
    # With articles
    "افتح ال{x}", "وريني ال{x}", "هات ال{x}",
    "افتح الملف بتاع {x}", "الملف بتاع {x} افتحه",
    # More natural
    "{x} فين", "{x} اللي حفظته", "{x} اللي نزلته",
    "{x} اللي عملته", "{x} اللي فاتحه",
    "وريني {x} اللي عندك", "افتح {x} اللي عندك",
]

# =============================================================================
# EXPANDED CODE-SWITCHING APP TEMPLATES (40+)
# =============================================================================
APP_TEMPLATES_MIXED = [
    "افتح ال{x}", "شغل ال{x}", "open ال{x}", "run ال{x}",
    "start ال{x}", "launch ال{x}", "افتح {x} app",
    "عايز افتح {x}", "ممكن تفتح ال{x}", "open {x} app",
    "شغل {x} بقى", "open {x} يا سيدي", "افتح {x} من عندك",
    "launch ال{x}", "ابدأ {x}", "افتح {x} على طول",
    "run {x} بسرعة", "start {x} دلوقتي", "open {x} please",
    "افتح {x} now", "شغل {x} now", "open {x} quickly",
    "عايز ال{x} يشتغل", "ممكن تشغل ال{x}", "can you open ال{x}",
    "ال{x} افتحه", "ال{x} شغله", "{x} شغلهولي",
    "open ال{x} بسرعة", "افتح ال{x} please", "run ال{x} عشاني",
    "start {x} يا عم", "launch {x} من فضلك", "open {x} عايزه",
    "ال{x} محتاجه تشتغل", "محتاج ال{x} يشتغل", "عايز افتح ال{x}",
    "مش عارف افتح ال{x}", "افتحلي ال{x}", "شغللي ال{x}",
]

# =============================================================================
# EXPANDED CODE-SWITCHING FILE TEMPLATES (40+)
# =============================================================================
FILE_TEMPLATES_MIXED = [
    "افتح ملف ال{x}", "open file ال{x}", "وريني ال{x}", "show me ال{x}",
    "عايز اشوف ال{x}", "افتح ال{x} ده", "open الملف {x}", "find ال{x}",
    "هات ال{x}", "pull up ال{x}", "افتح {x} file", "وريني {x} ده",
    "show me {x} file", "open {x} من فضلك", "وريني {x} بسرعة",
    "افتح {x} file ده", "open file {x} ده", "find {x} file",
    "هاتلي ال{x}", "جيب ال{x}", "get me ال{x}",
    "where is ال{x}", "فين ال{x}", "فين ملف ال{x}",
    "open ال{x} اللي على الديسكتوب", "افتح {x} اللي حفظته",
    "show me ال{x} اللي عندك", "وريني {x} اللي نزلته",
    "open latest {x}", "افتح اخر {x}", "open recent {x}",
    "افتح {x} اللي فاتحه", "open {x} اللي عملته",
    "هات {x} file", "جيبلي {x} file", "show ال{x}",
    "افتح ال{x} please", "open ال{x} بسرعة", "find {x} ده",
]

# =============================================================================
# MASSIVELY EXPANDED HAND-WRITTEN ENGLISH
# =============================================================================
HAND_WRITTEN_EN = [
    # Apps - casual
    ("hey can u open notepad real quick", "OPEN_APP", "notepad"),
    ("yo open chrome", "OPEN_APP", "chrome"),
    ("i need excel open like now", "OPEN_APP", "excel"),
    ("open up that calculator thing", "OPEN_APP", "calculator"),
    ("spotify pls", "OPEN_APP", "spotify"),
    ("can u start cmd for me", "OPEN_APP", "cmd"),
    ("get vlc running", "OPEN_APP", "vlc"),
    ("i wanna paint something open paint", "OPEN_APP", "paint"),
    ("settings, open them", "OPEN_APP", "settings"),
    ("terminal now please", "OPEN_APP", "terminal"),
    ("bruh open word", "OPEN_APP", "word"),
    ("need explorer asap", "OPEN_APP", "explorer"),
    ("quick open notepad", "OPEN_APP", "notepad"),
    ("chrome rn", "OPEN_APP", "chrome"),
    ("open calc pls", "OPEN_APP", "calculator"),
    ("fire up spotify", "OPEN_APP", "spotify"),
    ("gimme notepad", "OPEN_APP", "notepad"),
    ("pass me chrome", "OPEN_APP", "chrome"),
    ("boot up excel", "OPEN_APP", "excel"),
    ("turn on calculator", "OPEN_APP", "calculator"),
    # Apps - questions
    ("can you launch spotify", "OPEN_APP", "spotify"),
    ("could you open vlc", "OPEN_APP", "vlc"),
    ("is it possible to open paint", "OPEN_APP", "paint"),
    ("can we open settings", "OPEN_APP", "settings"),
    ("do you mind opening terminal", "OPEN_APP", "terminal"),
    # Apps - with filler
    ("okay open notepad", "OPEN_APP", "notepad"),
    ("alright launch chrome", "OPEN_APP", "chrome"),
    ("so start excel", "OPEN_APP", "excel"),
    ("just open calculator", "OPEN_APP", "calculator"),
    # Apps - urgency
    ("open spotify now", "OPEN_APP", "spotify"),
    ("need vlc quickly", "OPEN_APP", "vlc"),
    ("paint asap", "OPEN_APP", "paint"),
    ("settings immediately", "OPEN_APP", "settings"),
    # Apps - indirect
    ("i want notepad opened", "OPEN_APP", "notepad"),
    ("make chrome run", "OPEN_APP", "chrome"),
    ("get excel to work", "OPEN_APP", "excel"),
    ("i would like to open calculator", "OPEN_APP", "calculator"),
    # Apps - articles
    ("open the notepad", "OPEN_APP", "notepad"),
    ("launch that chrome", "OPEN_APP", "chrome"),
    ("start this excel", "OPEN_APP", "excel"),
    ("run the calculator app", "OPEN_APP", "calculator"),
    # Apps - slang/short
    ("notepad", "OPEN_APP", "notepad"),
    ("chrome", "OPEN_APP", "chrome"),
    ("excel", "OPEN_APP", "excel"),
    ("spotify", "OPEN_APP", "spotify"),
    ("vlc", "OPEN_APP", "vlc"),
    ("paint", "OPEN_APP", "paint"),
    ("cmd", "OPEN_APP", "cmd"),
    ("terminal", "OPEN_APP", "terminal"),
    ("settings", "OPEN_APP", "settings"),
    ("word", "OPEN_APP", "word"),
    ("explorer", "OPEN_APP", "explorer"),
    # Files - casual
    ("wheres my resume open it", "OPEN_FILE", "resume"),
    ("can u pull up the budget doc", "OPEN_FILE", "budget"),
    ("i need to see that report from yesterday", "OPEN_FILE", "report"),
    ("open my notes real fast", "OPEN_FILE", "notes"),
    ("show me that photo i saved", "OPEN_FILE", "photo"),
    ("wheres the presentation file", "OPEN_FILE", "presentation"),
    ("open that song i downloaded", "OPEN_FILE", "song"),
    ("pull the video up", "OPEN_FILE", "video"),
    ("my todo list open it", "OPEN_FILE", "todo"),
    ("open the project file", "OPEN_FILE", "project"),
    ("where is my resume", "OPEN_FILE", "resume"),
    ("display the budget", "OPEN_FILE", "budget"),
    ("need to check the report", "OPEN_FILE", "report"),
    ("show notes pls", "OPEN_FILE", "notes"),
    ("open that photo", "OPEN_FILE", "photo"),
    ("find presentation", "OPEN_FILE", "presentation"),
    ("play the song", "OPEN_FILE", "song"),
    ("show the video", "OPEN_FILE", "video"),
    ("open todo list", "OPEN_FILE", "todo"),
    ("access project file", "OPEN_FILE", "project"),
    # Files - questions
    ("where did i save the resume", "OPEN_FILE", "resume"),
    ("can you find the budget", "OPEN_FILE", "budget"),
    ("do you see the report", "OPEN_FILE", "report"),
    ("where is my notes file", "OPEN_FILE", "notes"),
    # Files - with location
    ("open resume from desktop", "OPEN_FILE", "resume"),
    ("show budget from downloads", "OPEN_FILE", "budget"),
    ("find report in documents", "OPEN_FILE", "report"),
    ("get notes from folder", "OPEN_FILE", "notes"),
    # Files - indirect
    ("i am looking for the photo", "OPEN_FILE", "photo"),
    ("i need the presentation", "OPEN_FILE", "presentation"),
    ("where did i put the song", "OPEN_FILE", "song"),
    ("i saved the video somewhere", "OPEN_FILE", "video"),
    # Files - articles/determiners
    ("open that resume", "OPEN_FILE", "resume"),
    ("show this budget", "OPEN_FILE", "budget"),
    ("find the report", "OPEN_FILE", "report"),
    ("open my recent notes", "OPEN_FILE", "notes"),
    # Files - slang/short
    ("resume", "OPEN_FILE", "resume"),
    ("budget", "OPEN_FILE", "budget"),
    ("report", "OPEN_FILE", "report"),
    ("notes", "OPEN_FILE", "notes"),
    ("photo", "OPEN_FILE", "photo"),
    ("presentation", "OPEN_FILE", "presentation"),
    ("song", "OPEN_FILE", "song"),
    ("video", "OPEN_FILE", "video"),
    ("todo", "OPEN_FILE", "todo"),
    ("project", "OPEN_FILE", "project"),
]

# =============================================================================
# MASSIVELY EXPANDED HAND-WRITTEN ARABIC
# =============================================================================
HAND_WRITTEN_AR = [
    # Apps - casual
    ("افتحلي الكروم بسرعة", "OPEN_APP", "chrome"),
    ("هوا فين النوت باد افتحه", "OPEN_APP", "notepad"),
    ("عايز الاكسل دلوقتي", "OPEN_APP", "excel"),
    ("شغللي السبوتيفاي", "OPEN_APP", "spotify"),
    ("ابعتلي الحاسبة", "OPEN_APP", "calculator"),
    ("فين الرسام افتحه", "OPEN_APP", "paint"),
    ("الاعدادات لو سمحت", "OPEN_APP", "settings"),
    ("عايز اشغل الترمينال", "OPEN_APP", "terminal"),
    ("افتحلي الوورد", "OPEN_APP", "word"),
    ("شغل الاكسبلورر", "OPEN_APP", "explorer"),
    ("عايز افتح الكوماند", "OPEN_APP", "cmd"),
    ("الفي ال سي شغله", "OPEN_APP", "vlc"),
    ("افتح الرسام بسرعة", "OPEN_APP", "paint"),
    ("الترمينال فين", "OPEN_APP", "terminal"),
    ("شغل السبوتيفاي", "OPEN_APP", "spotify"),
    # Apps - questions
    ("تقدر تفتح الكروم", "OPEN_APP", "chrome"),
    ("تعرف تشغل الاكسل", "OPEN_APP", "excel"),
    ("ممكن تفتحلي النوت باد", "OPEN_APP", "notepad"),
    ("تقدر تشغللي السبوتيفاي", "OPEN_APP", "spotify"),
    # Apps - casual filler
    ("يا عم افتح الكروم", "OPEN_APP", "chrome"),
    ("يا سيدي شغل الاكسل", "OPEN_APP", "excel"),
    ("طيب افتح النوت باد", "OPEN_APP", "notepad"),
    ("خلاص شغل السبوتيفاي", "OPEN_APP", "spotify"),
    # Apps - urgency
    ("افتح الكروم دلوقتي", "OPEN_APP", "chrome"),
    ("شغل الاكسل بسرعة", "OPEN_APP", "excel"),
    ("النوت باد على طول", "OPEN_APP", "notepad"),
    ("افتح السبوتيفاي دلوقتي", "OPEN_APP", "spotify"),
    # Apps - indirect
    ("عايز استخدم الكروم", "OPEN_APP", "chrome"),
    ("محتاج افتح الاكسل", "OPEN_APP", "excel"),
    ("محتاج النوت باد يشتغل", "OPEN_APP", "notepad"),
    ("مش عارف افتح السبوتيفاي", "OPEN_APP", "spotify"),
    # Apps - short/slang
    ("الكروم", "OPEN_APP", "chrome"),
    ("الاكسل", "OPEN_APP", "excel"),
    ("النوت باد", "OPEN_APP", "notepad"),
    ("السبوتيفاي", "OPEN_APP", "spotify"),
    ("الحاسبة", "OPEN_APP", "calculator"),
    ("الرسام", "OPEN_APP", "paint"),
    ("الاعدادات", "OPEN_APP", "settings"),
    ("الترمينال", "OPEN_APP", "terminal"),
    ("الوورد", "OPEN_APP", "word"),
    ("الاكسبلورر", "OPEN_APP", "explorer"),
    # Files - casual
    ("التقرير بتاعي فين افتحه", "OPEN_FILE", "report"),
    ("وريني الميزانية دلوقتي", "OPEN_FILE", "budget"),
    ("الصورة اللي حفظتها افتحها", "OPEN_FILE", "photo"),
    ("عايز اشوف العرض التقديمي بتاع الشغل", "OPEN_FILE", "presentation"),
    ("سيرتي الذاتية فين", "OPEN_FILE", "resume"),
    ("الاغنية اللي نزلتها افتحها", "OPEN_FILE", "song"),
    ("الفيديو بتاع امبارح افتحه", "OPEN_FILE", "video"),
    ("المهام بتاعتي وريني اياها", "OPEN_FILE", "todo"),
    ("افتح المشروع", "OPEN_FILE", "project"),
    ("وريني الملاحظات", "OPEN_FILE", "notes"),
    ("التقرير اللي على الديسكتوب", "OPEN_FILE", "report"),
    ("الصورة اللي محفوظة", "OPEN_FILE", "photo"),
    ("العرض التقديمي فين", "OPEN_FILE", "presentation"),
    ("السيرة الذاتية افتحها", "OPEN_FILE", "resume"),
    ("الاغنية دي افتحها", "OPEN_FILE", "song"),
    ("الفيديو ده", "OPEN_FILE", "video"),
    ("المهام", "OPEN_FILE", "todo"),
    ("مشروع الشغل", "OPEN_FILE", "project"),
    ("ميزانية الشهر", "OPEN_FILE", "budget"),
    # Files - questions
    ("فين التقرير", "OPEN_FILE", "report"),
    ("شوفت الميزانية", "OPEN_FILE", "budget"),
    ("لقيت الصورة", "OPEN_FILE", "photo"),
    ("فين العرض التقديمي", "OPEN_FILE", "presentation"),
    # Files - with location
    ("افتح التقرير اللي على الديسكتوب", "OPEN_FILE", "report"),
    ("وريني الميزانية اللي في الدونلودز", "OPEN_FILE", "budget"),
    ("الصورة اللي حفظتها فين", "OPEN_FILE", "photo"),
    # Files - indirect
    ("عايز اشوف التقرير", "OPEN_FILE", "report"),
    ("محتاج افتح الميزانية", "OPEN_FILE", "budget"),
    ("مش لاقي الصورة", "OPEN_FILE", "photo"),
    ("مش عارف لاقي العرض التقديمي", "OPEN_FILE", "presentation"),
    # Files - short
    ("التقرير", "OPEN_FILE", "report"),
    ("الميزانية", "OPEN_FILE", "budget"),
    ("الصورة", "OPEN_FILE", "photo"),
    ("العرض التقديمي", "OPEN_FILE", "presentation"),
    ("السيرة الذاتية", "OPEN_FILE", "resume"),
    ("الاغنية", "OPEN_FILE", "song"),
    ("الفيديو", "OPEN_FILE", "video"),
    ("المهام", "OPEN_FILE", "todo"),
    ("المشروع", "OPEN_FILE", "project"),
    ("الملاحظات", "OPEN_FILE", "notes"),
]

# =============================================================================
# MASSIVELY EXPANDED CODE-SWITCHING
# =============================================================================
HAND_WRITTEN_MIXED = [
    # Apps
    ("افتح الchrome", "OPEN_APP", "chrome"),
    ("open النوت باد", "OPEN_APP", "notepad"),
    ("عايز الexcel", "OPEN_APP", "excel"),
    ("شغل الspotify", "OPEN_APP", "spotify"),
    ("open الحاسبة", "OPEN_APP", "calculator"),
    ("الsettings افتحها", "OPEN_APP", "settings"),
    ("open التيرمينال", "OPEN_APP", "terminal"),
    ("افتح الcmd", "OPEN_APP", "cmd"),
    ("الvlc شغله", "OPEN_APP", "vlc"),
    ("open الرسام", "OPEN_APP", "paint"),
    ("افتح الword", "OPEN_APP", "word"),
    ("الexplorer افتحه", "OPEN_APP", "explorer"),
    ("run الchrome", "OPEN_APP", "chrome"),
    ("start الexcel", "OPEN_APP", "excel"),
    ("launch النوت باد", "OPEN_APP", "notepad"),
    ("افتح spotify بقى", "OPEN_APP", "spotify"),
    ("open الحاسبة please", "OPEN_APP", "calculator"),
    ("الsettings شغلها", "OPEN_APP", "settings"),
    ("run التيرمينال", "OPEN_APP", "terminal"),
    ("افتح cmd دلوقتي", "OPEN_APP", "cmd"),
    ("الvlc افتحه", "OPEN_APP", "vlc"),
    ("start الرسام", "OPEN_APP", "paint"),
    ("launch الword", "OPEN_APP", "word"),
    ("run الexplorer", "OPEN_APP", "explorer"),
    ("عايز افتح الchrome", "OPEN_APP", "chrome"),
    ("ممكن تشغل الexcel", "OPEN_APP", "excel"),
    ("افتحلي النوت باد", "OPEN_APP", "notepad"),
    ("شغللي الspotify", "OPEN_APP", "spotify"),
    ("open الcalculator", "OPEN_APP", "calculator"),
    ("افتح الsettings", "OPEN_APP", "settings"),
    ("open الterminal", "OPEN_APP", "terminal"),
    ("شغل الcmd", "OPEN_APP", "cmd"),
    ("افتح الvlc", "OPEN_APP", "vlc"),
    ("open الpaint", "OPEN_APP", "paint"),
    ("شغل الword", "OPEN_APP", "word"),
    ("افتح الexplorer", "OPEN_APP", "explorer"),
    # Files
    ("open التقرير", "OPEN_FILE", "report"),
    ("وريني الbudget", "OPEN_FILE", "budget"),
    ("الphoto دي افتحها", "OPEN_FILE", "photo"),
    ("افتح الpresentation", "OPEN_FILE", "presentation"),
    ("الresume فين", "OPEN_FILE", "resume"),
    ("open الاغنية", "OPEN_FILE", "song"),
    ("الvideo بتاع امبارح", "OPEN_FILE", "video"),
    ("افتح الtodo list", "OPEN_FILE", "todo"),
    ("open المشروع", "OPEN_FILE", "project"),
    ("الملاحظات وريني", "OPEN_FILE", "notes"),
    ("open الميزانية", "OPEN_FILE", "budget"),
    ("الreport على الديسكتوب", "OPEN_FILE", "report"),
    ("افتح الصورة", "OPEN_FILE", "photo"),
    ("الpresentation بتاع الشغل", "OPEN_FILE", "presentation"),
    ("open السيرة الذاتية", "OPEN_FILE", "resume"),
    ("الsong اللي نزلتها", "OPEN_FILE", "song"),
    ("افتح الفيديو", "OPEN_FILE", "video"),
    ("open المهام", "OPEN_FILE", "todo"),
    ("open file المشروع", "OPEN_FILE", "project"),
    ("وريني الnotes", "OPEN_FILE", "notes"),
    ("show me الreport", "OPEN_FILE", "report"),
    ("افتح file الbudget", "OPEN_FILE", "budget"),
    ("find الphoto", "OPEN_FILE", "photo"),
    ("get me الpresentation", "OPEN_FILE", "presentation"),
    ("where is الresume", "OPEN_FILE", "resume"),
    ("open الsong ده", "OPEN_FILE", "song"),
    ("افتح الvideo اللي حفظته", "OPEN_FILE", "video"),
    ("show me الtodo", "OPEN_FILE", "todo"),
    ("افتح الproject file", "OPEN_FILE", "project"),
    ("pull up الnotes", "OPEN_FILE", "notes"),
    ("open latest report", "OPEN_FILE", "report"),
    ("افتح اخر ميزانية", "OPEN_FILE", "budget"),
    ("open recent photo", "OPEN_FILE", "photo"),
    ("افتح presentation اللي فاتحه", "OPEN_FILE", "presentation"),
    ("get me resume file", "OPEN_FILE", "resume"),
    ("افتح song اللي نزلته", "OPEN_FILE", "song"),
    ("open video file", "OPEN_FILE", "video"),
    ("افتح todo list ده", "OPEN_FILE", "todo"),
    ("open project ده", "OPEN_FILE", "project"),
    ("افتح notes اللي عندك", "OPEN_FILE", "notes"),
]


def _stable_hash(s):
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (2**32)


def tokenize(text):
    """Tokenizer supporting Latin, Arabic, and mixed scripts."""
    text = re.sub(r"(ال)([a-z]+)", r" ", text, flags=re.IGNORECASE)
    return re.findall(r"[a-z0-9]+|[؀-ۿ]+", text.lower())


class Vocab:
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, texts):
        words = set()
        for t in texts:
            words.update(tokenize(t))
        self.words = [self.PAD, self.UNK] + sorted(words)
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}

    def __len__(self):
        return len(self.words)

    def encode(self, text, max_len):
        ids = [self.word_to_idx.get(w, 1) for w in tokenize(text)][:max_len]
        mask = [1.0] * len(ids)
        while len(ids) < max_len:
            ids.append(0)
            mask.append(0.0)
        return ids, mask

    def to_dict(self):
        return {"words": self.words}

    @staticmethod
    def from_dict(d):
        v = Vocab.__new__(Vocab)
        v.words = d["words"]
        v.word_to_idx = {w: i for i, w in enumerate(v.words)}
        return v


def _generate_for_lexicon(lexicon, templates_en, templates_ar, templates_mixed,
                          intent, has_command, n_per_surface=20):
    """Generate more examples per surface form (10 instead of 6)."""
    examples = []
    for canonical, value in lexicon.items():
        surface_forms = value[1] if has_command else value
        for surface in surface_forms:
            if is_arabic(surface):
                templates = templates_ar
            else:
                templates = templates_en

            k = min(n_per_surface, len(templates))
            chosen = np.random.default_rng(_stable_hash(surface)).choice(
                len(templates), size=k, replace=False
            )
            for i in chosen:
                examples.append({
                    "text": templates[int(i)].format(x=surface),
                    "intent": intent,
                    "target": canonical,
                    "lang": "ar" if is_arabic(surface) else "en",
                })

            # Mixed templates for all surfaces
            if templates_mixed:
                k_mixed = min(10, len(templates_mixed))  # More mixed examples
                chosen_mixed = np.random.default_rng(_stable_hash(surface + "_mixed")).choice(
                    len(templates_mixed), size=k_mixed, replace=False
                )
                for i in chosen_mixed:
                    examples.append({
                        "text": templates_mixed[int(i)].format(x=surface),
                        "intent": intent,
                        "target": canonical,
                        "lang": "mixed",
                    })
    return examples


def generate_examples(seed=0):
    examples = []

    # Generate with more examples per surface (10 instead of 6)
    examples += _generate_for_lexicon(
        APP_LEXICON, APP_TEMPLATES_EN, APP_TEMPLATES_AR, APP_TEMPLATES_MIXED,
        "OPEN_APP", has_command=True, n_per_surface=20
    )
    examples += _generate_for_lexicon(
        FILE_LEXICON, FILE_TEMPLATES_EN, FILE_TEMPLATES_AR, FILE_TEMPLATES_MIXED,
        "OPEN_FILE", has_command=False, n_per_surface=20
    )

    # Hand-written
    for text, intent, target in HAND_WRITTEN_EN:
        examples.append({"text": text, "intent": intent, "target": target, "lang": "en", "source": "hand_written"})
    for text, intent, target in HAND_WRITTEN_AR:
        examples.append({"text": text, "intent": intent, "target": target, "lang": "ar", "source": "hand_written"})
    for text, intent, target in HAND_WRITTEN_MIXED:
        examples.append({"text": text, "intent": intent, "target": target, "lang": "mixed", "source": "hand_written"})

    for e in examples:
        e.setdefault("source", "template")

    rng = np.random.default_rng(seed)
    rng.shuffle(examples)
    return examples


def build_dataset(max_len=12, seed=0):
    examples = generate_examples(seed=seed)
    texts = [e["text"] for e in examples]
    vocab = Vocab(texts)

    intents = ["OPEN_APP", "OPEN_FILE"]
    intent_to_idx = {name: i for i, name in enumerate(intents)}

    X_ids, X_mask, y = [], [], []
    for e in examples:
        ids, mask = vocab.encode(e["text"], max_len)
        X_ids.append(ids)
        X_mask.append(mask)
        y.append(intent_to_idx[e["intent"]])

    n_ar = sum(1 for e in examples if e["lang"] == "ar")
    n_en = sum(1 for e in examples if e["lang"] == "en")
    n_mixed = sum(1 for e in examples if e["lang"] == "mixed")

    return {
        "X_ids": np.array(X_ids, dtype=np.int64),
        "X_mask": np.array(X_mask, dtype=np.float64),
        "y": np.array(y, dtype=np.int64),
        "texts": texts,
        "langs": np.array([e["lang"] for e in examples]),
        "sources": np.array([e["source"] for e in examples]),
        "vocab": vocab,
        "intents": intents,
        "max_len": max_len,
        "lang_counts": {"en": n_en, "ar": n_ar, "mixed": n_mixed},
    }


if __name__ == "__main__":
    data = build_dataset()
    print(f"{len(data['texts'])} examples | vocab size {len(data['vocab'])} | max_len {data['max_len']}")
    print(f"Language split: {data['lang_counts']}")
    with open("raw_examples.json", "w", encoding="utf-8") as f:
        json.dump([{"text": t, "y": int(yy)} for t, yy in zip(data["texts"], data["y"])], f, indent=2, ensure_ascii=False)
