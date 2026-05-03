import sqlite3
import pandas as pd
import os

def load_data(db_path="data/CompSciencePub.sqlite"):
    """
    Connects to the SQLite database and loads the joined table 
    containing abstracts, journals, keywords, subjects, and keyword plus.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please check the path.")

    conn = sqlite3.connect(db_path)
    
    # Base query for AcademicRecord, Abstract, and Journal Name
    base_query = """
        SELECT 
            ar.AcademicRecordID AS article_id,
            ara.AbstractText AS abstract,
            p.PublicationID AS journal_id,
            p.Name AS journal_name
        FROM AcademicRecord ar
        JOIN AcademicRecordAbstract ara 
            ON ar.AcademicRecordID = ara.AcademicRecordId
        JOIN Publication p 
            ON ar.PublicationId = p.PublicationID
    """
    df_base = pd.read_sql_query(base_query, conn)
    
    # Author Keywords
    kw_query = """
        SELECT 
            ark.AcademicRecordId AS article_id,
            ak.Name AS keyword
        FROM AcademicRecordKeyword ark
        JOIN AcademicKeyword ak
            ON ark.AcademicKeywordId = ak.AcademicKeywordID
    """
    df_kw = pd.read_sql_query(kw_query, conn)
    df_kw_grouped = df_kw.groupby("article_id")["keyword"].apply(lambda x: " ".join(x.dropna())).reset_index()
    df_kw_grouped.rename(columns={"keyword": "author_keywords"}, inplace=True)
    
    # Subjects
    sub_query = """
        SELECT 
            ars.AcademicRecordId AS article_id,
            s.NameEn AS subject
        FROM AcademicRecordSubject ars
        JOIN AcademicSubject s
            ON ars.AcademicSubjectId = s.AcademicSubjectID
    """
    df_sub = pd.read_sql_query(sub_query, conn)
    df_sub_grouped = df_sub.groupby("article_id")["subject"].apply(lambda x: " ".join(x.dropna())).reset_index()
    df_sub_grouped.rename(columns={"subject": "subjects"}, inplace=True)
    
    # Keyword Plus
    kwp_query = """
        SELECT 
            arkp.AcademicRecordId AS article_id,
            akp.Name AS keyword_plus
        FROM AcademicRecordKeywordPlus arkp
        JOIN AcademicKeywordPlus akp
            ON arkp.AcademicKeywordPlusId = akp.AcademicKeywordPlusID
    """
    df_kwp = pd.read_sql_query(kwp_query, conn)
    df_kwp_grouped = df_kwp.groupby("article_id")["keyword_plus"].apply(lambda x: " ".join(x.dropna())).reset_index()
    
    # Merge all into main dataframe
    df = df_base.merge(df_kw_grouped, on="article_id", how="left")
    df = df.merge(df_sub_grouped, on="article_id", how="left")
    df = df.merge(df_kwp_grouped, on="article_id", how="left")
    
    # Fill NAs
    df.fillna("", inplace=True)
    
    # Create combined_text
    df["combined_text"] = (
        df["abstract"] + " " +
        df["author_keywords"] + " " +
        df["subjects"] + " " +
        df["keyword_plus"]
    )
    
    conn.close()
    return df

if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(df.head())
