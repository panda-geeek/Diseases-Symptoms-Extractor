package Diseases_Symptoms_Extractor;

import java.util.Arrays;
import java.util.Collection;
import java.util.Properties;
import org.javatuples.Triplet;
import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.naturalli.NaturalLogicAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

/**
 * Diseases and Symptoms Extractor
 *
 */
public class DiseasesSymptomsExtractor {
	public static void main(String[] args) throws Exception {

		diseaseSymptomsExtractor();
	}

	public static void diseaseSymptomsExtractor() throws Exception {
		String[] relationDict = { "cause", "symptom", "reason", "indication", "sign" };
		String[] diseasesDict = { "arrhythmia", "congenital heart disease", "heart attack", "heart valve disease",
				"cardiomyopathy" };
		//Creates a StanfordCoreNLP Pipeline, with Tokenization, Sentence Split, POS tagging, lemmatization, parsing and Open Information Extraction (OpenIE) annotators 
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma,depparse,natlog,openie");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		//Create an empty Annotation just with the given input text
		Annotation doc = new Annotation(
				"Heart disease is the numero uno killer globally, with nearly 17 million people falling prey to it in a year."
				+ " Heart diseases accounts for more deaths than all forms of cancer combined together."
				+ " Unhealthy lifestyle and eating habits are one of the major causes of heart diseases. "
				+ "Here are the top 5 most common and serious heart diseases along with its symptoms -"
				+ " Arrhythmia is a disorder of the heart rate wherein the heart beats too fast, too slow, or irregularly. "
				+ "Arrhythmia causes Fainting, Chest Pain. Second one is Congenital heart disease. "
				+ "Congenital heart disease is a complication with the heart’s structure and function at birth. "
				+ "Excessive sweating is the symptoms for Congenital heart disease. "
				+ "Third one is Heart attack. Heart attack is a permanent damage to the heart muscle and death of tissues due to lack of blood supply."
				+ " A heart attack usually occurs when a blood clot blocks blood flow to the heart. Sometime nausea is the reason for heart attack."
				+ " Fourth one is Heart valve disease. Heart valve disease occurs when one or more of the heart valves malfunction."
				+ " The main symptom of heart valve disease is an unusual heartbeat sound known as the heart murmur. "
				+ "Also, Fatigue is the indication for Heart valve disease. Last one is Cardiomyopathies."
				+ " Cardiomyopathies covers diseases of the heart muscle. People with these, sometimes called an enlarged heart, have hearts that are abnormally big, thickened, or stiffened."
				+ " Often, they lead to heart failure and abnormal heart rhythms. Trouble breathing is the sign for Cardiomyopathies.");
		
		// Run all Annotators defined in the pipeline on above text
		pipeline.annotate(doc);
		// Process each sentence in the document
		for (CoreMap sentence : doc.get(CoreAnnotations.SentencesAnnotation.class)) {
			// Get the OpenIE triples for the sentence
			Collection<RelationTriple> triples = sentence.get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);
			for (RelationTriple triple : triples) {
				//Checking if relation matching with any relation in our relation dictionary
				boolean isRelationFound = Arrays.stream(relationDict).parallel()
						.anyMatch(triple.relationLemmaGloss().toLowerCase()::contains);
				//Checking if Disease as subject matching with any Disease in our Disease dictionary
				boolean isDiseaseSubFound = Arrays.stream(diseasesDict).parallel()
						.anyMatch(triple.subjectLemmaGloss().toLowerCase()::contains);
				//Checking if Disease as object matching with any Disease in our Disease dictionary
				boolean isDiseaseObjFound = Arrays.stream(diseasesDict).parallel()
						.anyMatch(triple.objectLemmaGloss().toLowerCase()::contains);
				//Checking the Condition: Relation in the triplet must be match with any of our Relation Dictionary and either of Disease as subject or object match with any of the Disease in our Disease dictionary
				if (isRelationFound && (isDiseaseSubFound || isDiseaseObjFound)) {
					String[] subRelObj = { triple.subjectLemmaGloss(), triple.relationLemmaGloss(),triple.objectLemmaGloss() };
					Triplet<String, String, String> triplet = Triplet.fromArray(subRelObj);
					//Printing Matched Triplet and it's Coresponding Sentence 
					System.out.println("Sentence: " + sentence);
					System.out.println("Triplet: " + triplet + "\n");
				}
			}
		}
	}
}
