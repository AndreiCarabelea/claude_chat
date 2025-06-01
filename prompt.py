def get_long_prompt(lecture_text):

    prompt = f"""
    You are a highly skilled language model tasked with organizing and structuring a university lecture. The lecture covers various topics and needs to be divided into appropriate chapters and sections. 
    The lecture content is the following <LECTURE>{lecture_text}</LECTURE>
    
    Here's the step-by-step process you should follow:
    0. Analyze and remove any mistakes from it . If the content is not clear enough add your own additional explanations.
    
    1. **Identify Main Topics:** Analyze the text and identify the main topics covered in the lecture. These main topics will serve as the chapters.
    
    2. **Identify Subtopics:** Within each main topic, identify subtopics or key points that will serve as sections within each chapter.
    
   
    
    4. **Organize the Text:** Structure the text into chapters and sections according to the identified topics and subtopics. Ensure that each chapter starts on a new page.
    
    ### Example Output Structure:
    
    
    
    **Chapters and Sections**
    
    **Chapter 1: [Chapter Title]**
    
    [Text for Chapter 1]
    
    **Section 1: [Section Title]**
    
    [Text for Section 1]
    
    **Section 2: [Section Title]**
    
    [Text for Section 2]
    
    ...
    
    **Chapter 2: [Chapter Title]**
    
    [Text for Chapter 2]
    
    **Section 1: [Section Title]**
    
    [Text for Section 1]
    
    ...
    
    
    Your task is to transform the provided lecture text into the organized structure outlined above,  and also populate each chapter. Make sure to include all the information from the lecture text. Also make sure the formulas are formatted correctly.

    """
    return prompt