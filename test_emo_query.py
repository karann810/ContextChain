from emo import EpisodicMemoryObject, AgentEntry, RejectedOption


def test_rejected_compliance_query():
    emo = EpisodicMemoryObject(task_id='T1', raw_input='input')

    # Agent A rejects vendor X for compliance (compliance_gap=True)
    r1 = RejectedOption(name='VendorX', reason='Missing SOC2', risk_score=0.9, compliance_gap=True)
    # Agent B rejects vendor Y for performance (compliance_gap=False)
    r2 = RejectedOption(name='VendorY', reason='Slow I/O', risk_score=0.6, compliance_gap=False)

    emo.append(AgentEntry(agent_id='a', decision='d1', rejected_alternatives=[r1]))
    emo.append(AgentEntry(agent_id='b', decision='d2', rejected_alternatives=[r2]))

    results = emo.query('rejected_alternatives', 'compliance_gap == True')
    assert len(results) == 1
    assert results[0].name == 'VendorX'


if __name__ == '__main__':
    test_rejected_compliance_query()
    print('OK')
