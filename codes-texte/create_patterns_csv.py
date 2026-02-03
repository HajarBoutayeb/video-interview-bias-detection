import pandas as pd

patterns_data = [
    (r"(?i)\bwhere are you from\b", "Nationality/Ethnicity Bias", "general"),
    (r"(?i)\byou're.*\b(chinese|indian|african|arab|european)\b", "Nationality/Ethnicity Bias", "general"),
    (r"(?i)\bbecause.*\b(chinese|indian|african|arab)\b", "Nationality/Ethnicity Bias", "general"),
    (r"(?i)\byour english is.*\b(good|bad)\b", "Nationality/Ethnicity Bias", "general"),
    (r"(?i)\bdo you speak\b.*", "Nationality/Ethnicity Bias", "general"),
    (r"(?i)\bwe prefer.*\bnative speaker\b", "Nationality/Ethnicity Bias", "general"),

    (r"(?i)\byou have young children\b", "Gender Bias", "general"),
    (r"(?i)\bas a mother\b", "Gender Bias", "general"),
    (r"(?i)\bwe need someone stronger\b", "Gender Bias", "general"),
    (r"(?i)\bare you comfortable.*\b(travelling|working late)\b", "Gender Bias", "general"),
    (r"(?i)\bfor a woman\b", "Gender Bias", "general"),
    (r"(?i)\bfor a man\b", "Gender Bias", "general"),

    (r"(?i)\ba bit of age\b", "Age Bias", "general"),
    (r"(?i)\bsomeone older\b", "Age Bias", "general"),
    (r"(?i)\bsomeone younger\b", "Age Bias", "general"),
    (r"(?i)\byou're too young\b", "Age Bias", "general"),
    (r"(?i)\byou're too old\b", "Age Bias", "general"),
    (r"(?i)\bat your age\b", "Age Bias", "general"),

    (r"(?i)\byou look\b.*", "Appearance Bias", "general"),
    (r"(?i)\byour hairstyle\b", "Appearance Bias", "general"),
    (r"(?i)\byour dress\b", "Appearance Bias", "general"),
    (r"(?i)\btoo casual\b", "Appearance Bias", "general"),
    (r"(?i)\btoo formal\b", "Appearance Bias", "general"),

    (r"(?i)\bhere in\b.*\bwe\b.*", "Cultural Bias", "general"),
    (r"(?i)\bin our culture\b", "Cultural Bias", "general"),
    (r"(?i)\bwe don't do that here\b", "Cultural Bias", "general"),
    (r"(?i)\bwe expect.*\btradition\b", "Cultural Bias", "general"),
    (r"(?i)\byou should adapt to.*\bculture\b", "Cultural Bias", "general"),

    (r"(?i)\byour accent is\b.*", "Accent Bias", "general"),
    (r"(?i)\bhard to understand accent\b", "Accent Bias", "general"),
    (r"(?i)\bclear english\b", "Accent Bias", "general"),
    (r"(?i)\bwe prefer.*\baccent\b", "Accent Bias", "general"),

    (r"(?i)\byou studied at\b.*", "Education Bias", "general"),
    (r"(?i)\bwe prefer graduates from\b.*", "Education Bias", "general"),
    (r"(?i)\bonly top university\b", "Education Bias", "general"),
    (r"(?i)\bnot from a good school\b", "Education Bias", "general"),

    (r"(?i)\btoo confident\b", "Personality Bias", "general"),
    (r"(?i)\bnot confident enough\b", "Personality Bias", "general"),
    (r"(?i)\btoo quiet\b", "Personality Bias", "general"),
    (r"(?i)\btoo loud\b", "Personality Bias", "general"),

    (r"(?i)\bhe's like us\b", "Affinity Bias", "general"),
    (r"(?i)\bshe's not like us\b", "Affinity Bias", "general"),
    (r"(?i)\bknows the department\b", "Affinity Bias", "general"),
    (r"(?i)\bfrom the same town\b", "Affinity Bias", "general"),

    (r"(?i)\bwe need a man for this role\b", "Direct Discrimination", "general"),
    (r"(?i)\bwe need a woman for this role\b", "Direct Discrimination", "general"),

    (r"(?i)\byou might not fit in\b", "Exclusionary Language", "general"),
    (r"(?i)\bwe already have someone like you\b", "Tokenism", "general"),
]

video1_patterns = [
    (r"(?i)\bfrom china\b", "Nationality/Ethnicity Bias", "video1"),
    (r"(?i)\byou're chinese\b", "Nationality/Ethnicity Bias", "video1"),
    (r"(?i)\bbeing chinese\b", "Nationality/Ethnicity Bias", "video1"),
    (r"(?i)\bdoesn't speak french\b", "Cultural Bias", "video1"),
    (r"(?i)\bspeak french\b", "Cultural Bias", "video1"),
    (r"(?i)\bhere in switzerland\b", "Cultural Bias", "video1"),
    (r"(?i)\bi don't really speak french\b", "Language Bias", "video1"),
    (r"(?i)\bcan you speak french\b", "Language Bias", "video1"),
    (r"(?i)\bspeak english\b", "Language Bias", "video1"),
    (r"(?i)\bwith a bit of age\b", "Age Bias", "video1"),
    (r"(?i)\bwhat's the guy\b", "Appearance Bias", "video1"),
    (r"(?i)\bmiss tazin\b", "Name Bias", "video1"),
    (r"(?i)\bmr\.\s*z\b", "Name Bias", "video1"),
    (r"(?i)\bgood at math\b", "Stereotype Bias", "video1"),
    (r"(?i)\bexperienced entry-level\b", "Experience/Qualification Bias", "video1"),
    (r"(?i)\bmy mum told me\b", "Overconfidence Bias", "video1"),
    (r"(?i)\bwhat the f+\b", "Unprofessional Conduct", "video1"),
    (r"(?i)\bcan i go to the toilet\b", "Unprofessional Conduct", "video1"),
    (r"(?i)\bit was not so good\b", "Assumption Bias", "video1"),
    (r"(?i)\bi was holding a bias\b", "Self-Reflection Bias Acknowledgment", "video1"),
    (r"(?i)\bi shouldn't hold cultural bias\b", "Cultural Bias", "video1"),
    (r"(?i)\bwe should read their resumes\b", "Procedural Fairness", "video1"),
    (r"(?i)\bbiases are completely unacceptable\b", "Anti-Bias Policy", "video1"),
    (r"(?i)\bavoid any biases\b", "General Bias", "video1"),
]

video2_patterns = [
    (r"(?i)\btypes of biases\b", "General Bias", "video2"),
    (r"(?i)\bstereotyping\b", "General Bias", "video2"),
    (r"(?i)\bjackie chan\b", "Nationality/Ethnicity Bias", "video2"),
    (r"(?i)\bi'?m from china\b", "Nationality/Ethnicity Bias", "video2"),
    (r"(?i)\basian\b", "Nationality/Ethnicity Bias", "video2"),
    (r"(?i)\bstereotype of asians being good in math\b", "Stereotype Bias", "video2"),
    (r"(?i)\bbrought you a present\b", "Halo Effect Bias", "video2"),
    (r"(?i)\bdelighted to have you\b", "Halo Effect Bias", "video2"),
    (r"(?i)\bsince you're so generous\b", "Halo Effect Bias", "video2"),
    (r"(?i)\bgift\b", "Halo Effect Bias", "video2"),
    (r"(?i)\bcocky cheerleader\b", "Gender Bias", "video2"),
    (r"(?i)\bhorn effect\b", "Horn Effect Bias", "video2"),
    (r"(?i)\bgood-looking nature\b", "Horn Effect Bias", "video2"),
    (r"(?i)\bbaby\b", "Gender Bias", "video2"),
    (r"(?i)\byou can speak chinese\b", "Language Bias", "video2"),
    (r"(?i)\bspeak chinese\b", "Language Bias", "video2"),
    (r"(?i)\bi like chinese food\b", "Affinity Bias", "video2"),
    (r"(?i)\bpart of my family\b", "Affinity Bias", "video2"),
    (r"(?i)\bbig brother\b", "Affinity Bias", "video2"),
    (r"(?i)\btaking care of you\b", "Affinity Bias", "video2"),
    (r"(?i)\bno more interviews\b", "Procedural Unfairness", "video2"),
]

video3_patterns = [
    (r"(?i)\bperformance management\b", "Potential Bias", "video3"),
    (r"(?i)\bisaiah went to coppin\b", "Similar-to-me Bias", "video3"),
    (r"(?i)\balma mater\b", "Similar-to-me Bias", "video3"),
    (r"(?i)\bbig fit personality\b", "Similar-to-me Bias", "video3"),
    (r"(?i)\bdo really good here\b", "Similar-to-me Bias", "video3"),
    (r"(?i)\baliyah went to towson\b", "Experience Bias", "video3"),
    (r"(?i)\bdidn'?t have experience\b", "Experience Bias", "video3"),
    (r"(?i)\bpush him forward to the next round\b", "Procedural Unfairness", "video3"),
    (r"(?i)\bcoppin alum\b", "Affinity Bias", "video3"),
    (r"(?i)\bsame classes\b", "Affinity Bias", "video3"),
    (r"(?i)\bcaliber of people that graduate\b", "Education Bias", "video3"),
    (r"(?i)\bqualifications versus the school\b", "Education Bias", "video3"),
    (r"(?i)\bpersonality was good\b", "Halo Effect Bias", "video3"),
    (r"(?i)\bhaven't had some good experiences with people that went to towson\b", "School Bias / Negative Stereotype", "video3"),
    (r"(?i)\bnot to have our biases show up\b", "Anti-Bias Policy", "video3"),
    (r"(?i)\bqualifications\b", "Procedural Fairness", "video3"),
    (r"(?i)\bnot qualified to do the job\b", "Procedural Fairness", "video3"),
]
video4_patterns = [
    (r"(?i)\bare you from iran\b", "Nationality/Ethnicity Bias", "video4"),
    (r"(?i)\bhave you been to iran\b", "Nationality/Ethnicity Bias", "video4"),
    (r"(?i)\biran\b", "Nationality/Ethnicity Bias", "video4"),
    (r"(?i)\bwhat do you know about sales\b", "Experience / Qualification Bias", "video4"),
    (r"(?i)\byou are new to the pen market\b", "Experience / Qualification Bias", "video4"),
    (r"(?i)\btransferring emotions\b", "Procedural Fairness", "video4"),
    (r"(?i)\btransferring certainty\b", "Procedural Fairness", "video4"),
    (r"(?i)\bfavorite restaurant\b", "Appearance / Preference Bias", "video4"),
    (r"(?i)\blook at this face\b", "Appearance Bias + Unprofessional Conduct", "video4"),
    (r"(?i)\bbe beautiful\b", "Appearance Bias + Unprofessional Conduct", "video4"),
    (r"(?i)\bfulfilling full potential\b", "Potential Bias", "video4"),
    (r"(?i)\bsound fair enough\b", "Potential Bias", "video4"),
    (r"(?i)\bmy sister\b", "Nepotism / Cronyism Bias", "video4"),
    (r"(?i)\bposition for my sister\b", "Nepotism / Cronyism Bias", "video4"),
    (r"(?i)\bsister wants a job\b", "Nepotism / Cronyism Bias", "video4"),
    (r"(?i)\bwe'll be accepted\b", "Nepotism / Cronyism Bias", "video4"),
    (r"(?i)\byou will be accepted\b", "Nepotism / Cronyism Bias", "video4"),
    (r"(?i)\bi hired you\b", "Nepotism / Cronyism Bias", "video4"),
    (r"(?i)\bcousin\b", "Nepotism / Cronyism Bias", "video4"),
]

video5_patterns = [
    (r"(?i)\bdiversity, equity, and inclusion\b", "Cultural Bias", "video5"),
    (r"(?i)\bbiases in the selection process\b", "Procedural Fairness", "video5"),
    (r"(?i)\bstructured interviews\b", "Procedural Fairness", "video5"),
    (r"(?i)\bsame questions\b", "Procedural Fairness", "video5"),
    (r"(?i)\bculture fit\b", "Affinity Bias", "video5"),
    (r"(?i)\bcan.t really see myself going out to a bar\b", "Affinity Bias", "video5"),
    (r"(?i)\bhaving a beer\b", "Affinity Bias", "video5"),
    (r"(?i)\bselect candidates that look like them\b", "Affinity Bias", "video5"),
    (r"(?i)\bhave things in common\b", "Affinity Bias", "video5"),
    (r"(?i)\bwe know who we want\b", "Cultural Fit / Social Bias", "video5"),
    (r"(?i)\boffer him the role\b", "Cultural Fit / Social Bias", "video5"),
    (r"(?i)\bmore of a good culture fit\b", "Cultural Fit / Social Bias", "video5"),
    (r"(?i)\bwant to go out with after work\b", "Cultural Fit / Social Bias", "video5"),
]

video6_patterns = [
    (r"(?i)\bscottish\b", "Nationality/Ethnicity Bias", "video6"),
    (r"(?i)\bOch, I the noo\b", "Nationality/Ethnicity Bias", "video6"),
    (r"(?i)\bshe looks really strong\b", "Appearance Bias", "video6"),
    (r"(?i)\bcan we help you\b", "Appearance Bias", "video6"),
    (r"(?i)\btaking all the right boxes\b", "Personality Bias", "video6"),
    (r"(?i)\bshona. I'm sorry\b", "Assumption Bias", "video6"),
]

video7_patterns = [
    (r"(?i)\breally good education\b", "Experience / Qualification Bias", "video7"),
    (r"(?i)\bskills and qualifications\b", "Experience / Qualification Bias", "video7"),
    (r"(?i)\bdifferent situation than when they see you\b", "Racial Bias / Discrimination", "video7"),
    (r"(?i)\bcolor of my skin\b", "Race / Ethnicity Bias", "video7"),
    (r"(?i)\byou people tend to have a different concept of time\b", "Racial / Cultural Bias", "video7"),
    (r"(?i)\bjust not trying hard enough\b", "Bias Denial", "video7"),
    (r"(?i)\bworld isn't as fair as you think\b", "Bias Denial + Lived Experience Dismissal", "video7"),
    (r"(?i)\bI ask tough questions to see how people will handle pressure\b", "Bias Denial + Racism Justification", "video7"),
    (r"(?i)\bbeing judged before I even open my mouth\b", "Procedural Fairness", "video7"),
    (r"(?i)\bassess my abilities without bringing my race into it\b", "Procedural Fairness", "video7"),
    (r"(?i)\bsitting around and complaining\b", "Victim Blaming", "video7"),
    (r"(?i)\bsimilar histories\b", "Stereotyping / Assumptions", "video7"),
    (r"(?i)\bwon't crumble when they're slightly inconvenienced\b", "Stereotype Bias + Race Bias", "video7"),
    (r"(?i)\bperpetuating harmful stereotypes\b", "Stereotype Bias", "video7"),
    (r"(?i)\bunderstanding, Clark, this is discrimination\b", "Discrimination / Stereotyping", "video7"),
    (r"(?i)\bmaking assumptions based on my race\b", "Race / Ethnicity Bias", "video7"),
    (r"(?i)\bracist assumptions\b", "Racist bias", "video7"),
    (r"(?i)\bknown you were Mr. Callahan's son\b", "Status Bias", "video7"),
    (r"(?i)\bdo better next time\b", "Minimizing Racist Behavior", "video7")
]
video8_patterns = [
    (r"(?i)\bseriously\? you think you can get this job looking like that\b", "Appearance Bias", "video8"),
    (r"(?i)\bwhat skills could you possibly have\b", "Experience / Qualification Bias", "video8"),
    (r"(?i)\bI have over 20 years of experience\b", "Potential Bias", "video8"),
    (r"(?i)\bIt's all about how you manage it\b", "Potential Bias", "video8"),
    (r"(?i)\bbased on appearances, but real leaders\b", "Potential Bias", "video8"),
    (r"(?i)\bhandled demanding roles before\b", "Potential Bias", "video8"),
    (r"(?i)\btime management and staying focused\b", "Potential Bias", "video8"),
    (r"(?i)\bmanaged several large-scale projects\b", "Potential Bias", "video8"),
    (r"(?i)\bcan you handle the pressure\b", "Stereotype Bias", "video8"),
    (r"(?i)\band you think they'll take you seriously with that cane\b", "Disability Bias", "video8"),
    (r"(?i)\bpeople will see weakness, not capability\b", "Disability Bias", "video8"),
    (r"(?i)\bhow do you think that your condition will affect\b", "Disability Bias", "video8"),
    (r"(?i)\bMy disability doesn't define\b", "Disability Bias", "video8"),
    (r"(?i)\bconsistently delivered high quality work despite\b", "Disability Bias", "video8"),
    (r"(?i)\bPhysical limitations\b", "Disability Bias", "video8"),
    (r"(?i)\bhigh-pressure environment\b", "Disability Bias", "video8"),
    (r"(?i)\bhandle the demands of the job\b", "Disability Bias", "video8"),
    (r"(?i)\bSorry, do you need help\b", "Disability Bias", "video8"),
    (r"(?i)\bDisability Bias \+ Unprofessional Conduct\b", "Disability Bias + Unprofessional Conduct", "video8"),
    (r"(?i)\bmy dad knows the CEO\b", "Nepotism / Cronyism Bias", "video8"),
    (r"(?i)\bWell, I bet you don't know her like my family does\b", "Nepotism/Cronyism Bias", "video8"),
    (r"(?i)\bMy dad has been friends with her for years\b", "Nepotism/Cronyism Bias", "video8"),
    (r"(?i)\bthanks to my family, I have a very extensive network\b", "Nepotism/Cronyism Bias", "video8"),
    (r"(?i)\bwith my father's connections\b", "Nepotism / Cronyism Bias", "video8"),
    (r"(?i)\bNepotism\b", "Nepotism", "video8"),
    (r"(?i)\bNepotism Bias \+ Disability Bias\b", "Nepotism Bias + Disability Bias", "video8"),
    (r"(?i)\bRespect is for people who can afford to lose\b", "Personality Bias", "video8"),
    (r"(?i)\bI can work in a team, but I prefer to lead\b", "Personality Bias", "video8"),
    (r"(?i)\bI find that I get better results when I work with people\b", "Dominance Bias", "video8"),
    (r"(?i)\bI'm the one in charge\b", "Dominance Bias", "video8"),
    (r"(?i)\bI'd make the final decision\b", "Dominance Bias \+ Overconfidence Bias", "video8"),
    (r"(?i)\bI'm pretty confident in my judgment\b", "Dominance Bias \+ Overconfidence Bias", "video8"),
    (r"(?i)\bHonestly, I don't have many\b", "Overconfidence Bias", "video8"),
    (r"(?i)\bI guess sometimes I can be a bit too demanding\b", "Overconfidence Bias", "video8"),
    (r"(?i)\bconstructive criticism is fine, but I know what I'm doing\b", "Overconfidence Bias", "video8"),
]

video9_patterns = [
    (r"(?i)\bnot the best fit for our curriculum\b", "Experience / Qualification Bias", "video9"),
    (r"(?i)\bobvious who's more senior\b", "Seniority Bias", "video9"),
    (r"(?i)\bdo you really think she had the right, erm, gravitas\b", "Academic Elitism Bias", "video9"),
    (r"(?i)\bor is gravitas more important\b", "Academic Elitism Bias", "video9"),
    (r"(?i)\bher Newcastle accent\b", "Accent Bias", "video9"),
    (r"(?i)\bused diversity and inclusion to justify her bias\b", "Justification Bias", "video9"),
    (r"(?i)\bhe knows the department\b", "Affinity Bias", "video9"),
    (r"(?i)\bthat's classic in-group thinking\b", "in-group bias", "video9"),
    (r"(?i)\bshe's not like us\b", "Affinity Bias", "video9"),
    (r"(?i)\bexcellent paper at the Geocom conference\b", "Halo Effect Bias", "video9"),
    (r"(?i)\bStacey's young for a lectureship\b", "Age Bias", "video9"),
    (r"(?i)\bRight. And she has young children\b", "Parental Status Bias", "video9"),
    (r"(?i)\bRelocation would be difficult\b", "Assumption Bias", "video9"),
    (r"(?i)\bneed someone more committed to their career\b", "Commitment Bias", "video9"),
    (r"(?i)\bSomeone older\b", "Age Bias", "video9"),
    (r"(?i)\bAnd that is direct discrimination\b", "age/gender discrimination", "video9"),
]


all_patterns = (
    patterns_data +
    video1_patterns +
    video2_patterns +
    video3_patterns +
    video4_patterns +
    video5_patterns +
    video6_patterns +
    video7_patterns +
    video8_patterns +
    video9_patterns
)

df = pd.DataFrame(all_patterns, columns=["Pattern", "Bias Type", "Source"])

color_map = {
    "Nationality/Ethnicity Bias": "#FFD700",
    "Gender Bias": "#FF69B4",
    "Age Bias": "#87CEEB",
    "Appearance Bias": "#FFA07A",
    "Cultural Bias": "#90EE90",
    "Accent Bias": "#9370DB",
    "Education Bias": "#20B2AA",
    "Personality Bias": "#FFB6C1",
    "Affinity Bias": "#CD5C5C",
    "Direct Discrimination": "#8B0000",
    "Exclusionary Language": "#708090",
    "Tokenism": "#F5DEB3",
    "Experience / Qualification Bias": "#ADD8E6",
    "Seniority Bias": "#B0C4DE",
    "Academic Elitism Bias": "#DAA520",
    "Justification Bias": "#D2691E",
    "Halo Effect Bias": "#FF8C00",
    "Parental Status Bias": "#4682B4",
    "Assumption Bias": "#2E8B57",
    "Commitment Bias": "#556B2F",
    "Overconfidence Bias": "#FF4500",
    "Disability Bias": "#8A2BE2",
    "Nepotism / Cronyism Bias": "#FF6347",
    "Superiority Complex": "#FF1493",
    "Condescension": "#A52A2A",
    "Unprofessional Conduct": "#800080",
    "Potential Bias": "#7FFF00"
}


def highlight_rows(row):
    color = color_map.get(row["Bias Type"], "#FFFFFF")
    return [f"background-color: {color}"] * len(row)

df_styled = df.style.apply(highlight_rows, axis=1)
df_styled.to_excel("bias_patterns.xlsx", index=False)

print("Le fichier bias_patterns.xlsx a été créé avec succès")