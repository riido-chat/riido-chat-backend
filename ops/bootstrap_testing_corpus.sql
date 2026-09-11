-- One-time remote-dev setup after Alembic 20260911_11.
-- Run this transaction against the intended database, then run:
--   python -m app.ops.bootstrap_testing_corpus
-- The application command copies generated-identity corpus rows in one
-- transaction and creates the target ACTIVE index. This SQL intentionally
-- never creates HELP_CHATBOT_TEST or any corpus rows.
BEGIN;

DO $$
DECLARE
    v_profile_id bigint;
    v_target_group_id bigint;
    v_testing_revision_id bigint;
    v_revision_count integer;
    v_next_version integer;
BEGIN
    SELECT id INTO v_profile_id
    FROM chat_profiles
    WHERE profile_key = 'HELP_CHATBOT';
    IF v_profile_id IS NULL THEN
        RAISE EXCEPTION 'HELP_CHATBOT chat profile is missing';
    END IF;

    SELECT id INTO v_target_group_id
    FROM document_groups
    WHERE group_key = 'HELP_CHATBOT_TEST';
    IF v_target_group_id IS NULL THEN
        RAISE EXCEPTION 'HELP_CHATBOT_TEST document group is missing; refusing to create it';
    END IF;

    SELECT count(*) INTO v_revision_count
    FROM chat_profile_revisions
    WHERE chat_profile_revisions.profile_id = v_profile_id AND status = 'TESTING';
    IF v_revision_count > 1 THEN
        RAISE EXCEPTION 'HELP_CHATBOT has more than one TESTING revision';
    END IF;

    IF v_revision_count = 1 THEN
        SELECT id INTO v_testing_revision_id
        FROM chat_profile_revisions
        WHERE chat_profile_revisions.profile_id = v_profile_id
          AND status = 'TESTING';
        UPDATE chat_profile_revisions
        SET document_group_id = v_target_group_id,
            generation_model_name = 'gpt-5.6-terra',
            generation_prompt_version = 'v24',
            query_rewrite_model_name = 'gpt-5.4-mini',
            query_rewrite_prompt_version = 'v7',
            semantic_cache_enabled = false
        WHERE id = v_testing_revision_id;
    ELSE
        SELECT coalesce(max(version), 0) + 1 INTO v_next_version
        FROM chat_profile_revisions
        WHERE chat_profile_revisions.profile_id = v_profile_id;
        INSERT INTO chat_profile_revisions
            (profile_id, version, document_group_id, status,
             generation_model_name, generation_prompt_version,
             query_rewrite_model_name, query_rewrite_prompt_version,
             semantic_cache_enabled)
        VALUES
            (v_profile_id, v_next_version, v_target_group_id, 'TESTING',
             'gpt-5.6-terra', 'v24', 'gpt-5.4-mini', 'v7', false);
    END IF;
END $$;

COMMIT;
