"""Export prediction provenance without asserting that scores are ground truth."""
from pathlib import Path
from urllib.parse import quote

def export_graph(predictions, output):
    from rdflib import Graph, Namespace, URIRef, Literal, RDF
    graph=Graph()
    onto=Namespace('https://w3id.org/ontoforensics/utterance#')
    prov=Namespace('http://www.w3.org/ns/prov#')
    sosa=Namespace('http://www.w3.org/ns/sosa/')
    graph.bind('uf',onto); graph.bind('prov',prov); graph.bind('sosa',sosa)
    for row in predictions:
        recording=URIRef('urn:recording:'+quote(row['recording_id'],safe=''))
        decision=URIRef(str(recording)+':decision')
        graph.add((recording,RDF.type,onto.Recording))
        graph.add((decision,RDF.type,onto.FusionDecision))
        graph.add((decision,prov.wasDerivedFrom,recording))
        graph.add((decision,onto.status,Literal(row['status'])))
        if row['probability'] is not None:
            graph.add((decision,onto.fakeProbability,Literal(row['probability'])))
        for item in row.get('evidence',[]):
            observation=URIRef(str(recording)+':expert:'+quote(item['expert'],safe=''))
            model=URIRef('urn:sha256:'+item['checkpoint_sha256'])
            graph.add((observation,RDF.type,sosa.Observation))
            graph.add((observation,sosa.hasFeatureOfInterest,recording))
            graph.add((observation,sosa.hasSimpleResult,Literal(item['score'])))
            graph.add((observation,prov.wasGeneratedBy,model))
            graph.add((model,onto.family,Literal(item['family'])))
            graph.add((observation,onto.configSHA256,Literal(item['config_sha256'])))
            graph.add((decision,prov.wasDerivedFrom,observation))
        for name,value in row.get('calibrated_contributions',{}).items():
            contribution=URIRef(str(decision)+':feature:'+quote(name,safe=''))
            graph.add((decision,onto.hasContribution,contribution))
            graph.add((contribution,onto.featureName,Literal(name)))
            graph.add((contribution,onto.logitContribution,Literal(value)))
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    graph.serialize(destination=str(output),format='turtle')
    return {'triples':len(graph),'recordings':len(predictions),'supervision_exported':False}
