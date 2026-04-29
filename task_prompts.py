"""
task_prompts.py — Task-specific descriptions and questions for LLM prompts.

Each RelBench task gets a human-readable description and a targeted question.
For unseen tasks (generalization), a fallback auto-generates prompts from the
task name and type.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Task descriptions: 1–2 sentences explaining what the task predicts and why
# ─────────────────────────────────────────────────────────────────────────────

TASK_DESCRIPTION = {
    # ── commerce-1 ──
    "diginetica-downsample-ctr": (
        "Predict whether a user will click on the recommended product based on "
        "their browsing session and product features."
    ),
    "rel-hm-item-sales": (
        "Predict the future sales volume of a fashion item based on its "
        "attributes, historical transactions, and customer demographics."
    ),
    "rel-hm-user-churn": (
        "Predict whether a customer of the fashion retailer will stop "
        "purchasing in the upcoming period based on their transaction history."
    ),
    "retailrocket-cvr": (
        "Predict whether a user session will result in a purchase (conversion) "
        "based on browsing behavior and product interactions."
    ),
    "seznam-charge": (
        "Predict the charge bucket (8-class) a user falls into on the "
        "advertising platform based on their campaign and account data."
    ),
    "seznam-prepay": (
        "Predict the prepayment bucket (8-class) for a user on the "
        "advertising platform based on historical billing and campaign data."
    ),

    # ── commerce-2 ──
    "amazon-churn": (
        "Predict whether a customer will churn (stop reviewing products) on "
        "the e-commerce platform based on their review and purchase history."
    ),
    "amazon-rating": (
        "Predict the star rating a customer will give to a product based on "
        "their past review patterns and product attributes."
    ),
    "outbrain-small-ctr": (
        "Predict whether a user will click on a content recommendation based "
        "on their reading history and the article features."
    ),
    "rel-avito-ad-ctr": (
        "Predict the click-through rate of an advertisement on the "
        "classifieds platform based on ad content and user context."
    ),
    "rel-avito-user-clicks": (
        "Predict whether a user will click on at least one ad in the upcoming "
        "period based on their browsing and search history."
    ),
    "rel-avito-user-visits": (
        "Predict whether a user will visit the classifieds platform in the "
        "upcoming period based on their historical engagement patterns."
    ),

    # ── others-1 ──
    "rel-f1-driver-dnf": (
        "Predict whether a Formula 1 driver will fail to finish (DNF) a race "
        "based on their career statistics, team data, and circuit conditions."
    ),
    "rel-f1-driver-position": (
        "Predict the finishing position of a Formula 1 driver in a race based "
        "on their historical performance, qualifying results, and team data."
    ),
    "rel-f1-driver-top3": (
        "Predict whether a Formula 1 driver will finish in the top 3 (podium) "
        "based on their season performance and race conditions."
    ),
    "stackexchange-churn": (
        "Predict whether a StackExchange user will become inactive (churn) "
        "based on their question, answer, and voting history."
    ),
    "stackexchange-upvote": (
        "Predict whether a StackExchange post will receive above-threshold "
        "upvotes based on content quality signals and author reputation."
    ),
    "virus-wnv-pred": (
        "Predict the presence of West Nile Virus in a mosquito trap based on "
        "geographic, weather, and historical surveillance data."
    ),

    # ── others-2 ──
    "airbnb-destination": (
        "Predict which country a new Airbnb user will book their first trip "
        "to based on their demographic and session data."
    ),
    "rel-trial-site-success": (
        "Predict whether a clinical trial site will successfully complete "
        "enrollment based on site characteristics and trial protocol data."
    ),
    "rel-trial-study-adverse": (
        "Predict the rate of adverse events in a clinical trial based on "
        "study design, drug characteristics, and patient population data."
    ),
    "rel-trial-study-outcome": (
        "Predict the primary outcome of a clinical trial based on study "
        "design, intervention type, and historical trial data."
    ),
    "talkingdata-demo-pred": (
        "Predict demographic attributes of a mobile user based on their app "
        "usage patterns and device information."
    ),
    "telstra-severity": (
        "Predict the severity level of a network fault based on event logs, "
        "resource allocation, and location data."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Task questions: the specific question posed to the LLM after context
# ─────────────────────────────────────────────────────────────────────────────

TASK_QUESTION = {
    # ── commerce-1 ──
    "diginetica-downsample-ctr": (
        "Based on the user session and product data provided, will the user "
        "click on this product? Answer Yes or No."
    ),
    "rel-hm-item-sales": (
        "Based on the item and transaction data provided, what will be the "
        "sales volume for this item? Output a single number."
    ),
    "rel-hm-user-churn": (
        "Based on the customer data provided, will this customer stop "
        "purchasing in the next period? Answer Yes or No."
    ),
    "retailrocket-cvr": (
        "Based on the session data provided, will this session result in a "
        "purchase? Answer Yes or No."
    ),
    "seznam-charge": (
        "Based on the account and campaign data provided, which charge bucket "
        "does this user fall into? Output the class index."
    ),
    "seznam-prepay": (
        "Based on the billing data provided, which prepayment bucket does "
        "this user fall into? Output the class index."
    ),

    # ── commerce-2 ──
    "amazon-churn": (
        "Based on the customer data provided, will this customer churn in the "
        "next period? Answer Yes or No."
    ),
    "amazon-rating": (
        "Based on the customer and product data provided, what star rating "
        "will the customer give? Output a single number."
    ),
    "outbrain-small-ctr": (
        "Based on the user reading history and article data provided, will the "
        "user click on this recommendation? Answer Yes or No."
    ),
    "rel-avito-ad-ctr": (
        "Based on the ad and user context provided, what is the predicted "
        "click-through rate for this ad? Output a single number."
    ),
    "rel-avito-user-clicks": (
        "Based on the user data provided, will this user click on at least "
        "one ad in the next period? Answer Yes or No."
    ),
    "rel-avito-user-visits": (
        "Based on the user data provided, will this user visit the platform "
        "in the next period? Answer Yes or No."
    ),

    # ── others-1 ──
    "rel-f1-driver-dnf": (
        "Based on the driver and race data provided, will this driver fail to "
        "finish the race? Answer Yes or No."
    ),
    "rel-f1-driver-position": (
        "Based on the driver's profile, team, and recent race history "
        "provided, what is the predicted finishing position? Output a single "
        "number."
    ),
    "rel-f1-driver-top3": (
        "Based on the driver and race data provided, will this driver finish "
        "in the top 3? Answer Yes or No."
    ),
    "stackexchange-churn": (
        "Based on the user activity data provided, will this user become "
        "inactive? Answer Yes or No."
    ),
    "stackexchange-upvote": (
        "Based on the post and author data provided, will this post receive "
        "above-threshold upvotes? Answer Yes or No."
    ),
    "virus-wnv-pred": (
        "Based on the trap location and environmental data provided, is West "
        "Nile Virus present? Answer Yes or No."
    ),

    # ── others-2 ──
    "airbnb-destination": (
        "Based on the user data provided, which destination country will this "
        "user book? Output the class index."
    ),
    "rel-trial-site-success": (
        "Based on the trial site data provided, what is the predicted site "
        "success score for enrollment completion? Output a single number."
    ),
    "rel-trial-study-adverse": (
        "Based on the study data provided, what will be the adverse event "
        "rate? Output a single number."
    ),
    "rel-trial-study-outcome": (
        "Based on the study data provided, will this study achieve a "
        "successful primary outcome? Answer Yes or No."
    ),
    "talkingdata-demo-pred": (
        "Based on the device and app usage data provided, what is the "
        "predicted demographic group? Output the class index."
    ),
    "telstra-severity": (
        "Based on the event and resource data provided, what is the predicted "
        "fault severity level? Output the class index."
    ),
}


def get_task_description(task_name: str) -> str:
    """Return task description, auto-generating for unseen tasks."""
    if task_name in TASK_DESCRIPTION:
        return TASK_DESCRIPTION[task_name]
    # Fallback for unseen tasks (generalization)
    clean = task_name.replace("-", " ").replace("_", " ")
    return f"Predict the target value for the '{clean}' task based on the entity's relational database record and connected entities."


def get_task_question(task_name: str, task_type: str) -> str:
    """Return task question, auto-generating for unseen tasks."""
    if task_name in TASK_QUESTION:
        return TASK_QUESTION[task_name]
    # Fallback for unseen tasks
    clean = task_name.replace("-", " ").replace("_", " ")
    if task_type == "regression":
        return f"Based on the data provided for '{clean}', what is the predicted value? Output a single number."
    else:
        return f"Based on the data provided for '{clean}', what is the predicted class? Output the class index."


def audit_task_prompts(metatask: dict, verbose: bool = True) -> list:
    """Cross-check that each task's prompt format matches its metatask.yaml type.

    For binary tasks (num_class==2) the question must end with "Yes or No".
    For regression (num_class==1) it must say "Output a single number".
    For multi-class (num_class>2) it must say "Output the class index".

    Returns the list of mismatches; prints them when verbose=True.
    Call this once at startup with the loaded metatask dict.
    """
    mismatches = []
    for tn, meta in metatask.items():
        if tn not in TASK_QUESTION:
            continue
        q = TASK_QUESTION[tn]
        num_class = meta.get("num_class", 1)
        task_type = meta.get("task_type", "regression")

        if task_type == "regression" or num_class == 1:
            expected = "single number"
        elif num_class == 2:
            expected = "yes or no"
        else:
            expected = "class index"

        if expected not in q.lower():
            mismatches.append((tn, num_class, task_type, expected, q))
            if verbose:
                print(f"[PROMPT AUDIT] MISMATCH on '{tn}': "
                      f"num_class={num_class}, task_type={task_type}, "
                      f"expected='{expected}' but question is:\n  {q}")

    if verbose and not mismatches:
        print("[PROMPT AUDIT] All task prompts match their metadata.")
    return mismatches


# ─────────────────────────────────────────────────────────────────────────────
# Rich metadata-aware prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def _clean_feat_name(feat_name: str) -> str:
    """Convert 'EntityType___Griffin_text_field' or 'EntityType___TIMESTAMP(x)'
    into a human-readable column description."""
    # Strip entity prefix: "airbnb-User___age" → "age"
    if "___" in feat_name:
        feat_name = feat_name.split("___", 1)[1]
    # Handle Griffin text features: "Griffin_text_language" → "language (text)"
    if feat_name.startswith("Griffin_text_"):
        return feat_name[len("Griffin_text_"):].replace("_", " ") + " (text)"
    # Handle timestamps: "TIMESTAMP(date_created)" → "date_created (timestamp)"
    if feat_name.startswith("TIMESTAMP(") and feat_name.endswith(")"):
        inner = feat_name[len("TIMESTAMP("):-1]
        return inner.replace("_", " ") + " (timestamp)"
    return feat_name.replace("_", " ")


def _clean_entity_name(entity_name: str) -> str:
    """Convert 'rel-hm-Article' → 'Article' or 'airbnb-User' → 'User'."""
    # Strip dataset prefix (everything before the last hyphen-separated
    # capitalised word).  E.g. "rel-hm-Article" → "Article",
    # "amazon-Review" → "Review", "stackexchange-User" → "User".
    parts = entity_name.split("-")
    # Walk backwards to find the first part that starts with uppercase
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] and parts[i][0].isupper():
            return "-".join(parts[i:]).replace("_", " ")
    return entity_name.replace("_", " ").replace("-", " ")


def _parse_edge_name(edge_name: str):
    """Parse 'head of Src:Src-fk_col:Dst' or 'tail of Src:Src-fk_col:Dst'
    into (direction, source_entity, fk_column, target_entity)."""
    direction = "outgoing" if edge_name.startswith("head of ") else "incoming"
    body = edge_name.removeprefix("head of ").removeprefix("tail of ")
    # body = "Src:Src-fk_col:Dst"
    colon_parts = body.split(":")
    if len(colon_parts) == 3:
        src_entity = colon_parts[0]
        # middle part: "Src-fk_col" — strip the entity prefix
        mid = colon_parts[1]
        fk_col = mid.split("-", 1)[1] if "-" in mid else mid
        dst_entity = colon_parts[2]
        return direction, src_entity, fk_col, dst_entity
    return direction, body, "", ""


def build_rich_system_prompt(
    task_name: str,
    task_type: str,
    root_entity: str,
    metanode: dict,
    metaadj: dict,
    metatask: dict,
    feature_names: list[str] | None = None,
    neighbor_entity_types: list[str] | None = None,
) -> str:
    """Build a metadata-rich system prompt describing the relational schema.

    Inspired by Rel-LLM's approach of giving the LLM contextual understanding
    of the relational structure.  This is critical for generalization to unseen
    datasets: instead of relying solely on GNN embeddings, the LLM also gets a
    textual description of the schema, relationships, and task semantics.

    Args:
        task_name:  Task identifier (e.g. "rel-f1-driver-position").
        task_type:  "regression" or "retrieval".
        root_entity: Target entity type (e.g. "rel-f1-Driver").
        metanode:   Graph.metanode dict — entity metadata.
        metaadj:    Adjacency metadata (in/out edges per entity).
        metatask:   Task.metatask dict — task metadata.
        feature_names:  Ordered list of feature names for the root entity
                        (after masking).  If None, derived from metanode.
        neighbor_entity_types: Entity types of neighbors present in the
                               subgraph.  Used to describe neighbor context.

    Returns:
        A multi-paragraph system prompt string.
    """
    task_meta = metatask.get(task_name, {})
    description = get_task_description(task_name)
    root_clean = _clean_entity_name(root_entity)

    # ── Section 1: Task overview ──
    lines = [
        "You are a predictive model for relational databases.",
        f"Task: {description}",
        f"Prediction target: {root_clean} entity.",
    ]
    if task_type == "regression":
        metric = task_meta.get("metric", "rmse")
        lines.append(f"This is a regression task (metric: {metric}).")
    else:
        num_class = task_meta.get("num_class", 2)
        metric = task_meta.get("metric", "retrieval_auroc")
        lines.append(
            f"This is a classification task with {num_class} classes "
            f"(metric: {metric})."
        )

    # ── Section 2: Target entity schema ──
    if feature_names is None and root_entity in metanode:
        feature_names = metanode[root_entity].get("feat", [])
    if feature_names:
        masked = set(task_meta.get("masked_feat", []))
        visible = [f for f in feature_names if f not in masked]
        clean_cols = [_clean_feat_name(f) for f in visible]
        lines.append("")
        lines.append(f"The {root_clean} entity has these attributes:")
        for col in clean_cols:
            lines.append(f"  - {col}")

    # ── Section 3: Relational context (relationships) ──
    if root_entity in metaadj or root_entity in metanode:
        adj_info = metaadj.get(root_entity, metanode.get(root_entity, {}))
        in_edges = adj_info.get("in", [])
        out_edges = adj_info.get("out", [])
        if in_edges or out_edges:
            lines.append("")
            lines.append(f"Relational context for {root_clean}:")
            seen = set()
            for e in out_edges:
                _, src, fk, dst = _parse_edge_name(e)
                dst_clean = _clean_entity_name(dst)
                desc = f"  - connects to {dst_clean} via '{fk.replace('_', ' ')}'"
                if desc not in seen:
                    lines.append(desc)
                    seen.add(desc)
            for e in in_edges:
                _, src, fk, dst = _parse_edge_name(e)
                src_clean = _clean_entity_name(src)
                desc = f"  - referenced by {src_clean} via '{fk.replace('_', ' ')}'"
                if desc not in seen:
                    lines.append(desc)
                    seen.add(desc)

    # ── Section 4: Neighbor entity descriptions (1-hop) ──
    one_hop_entities = set()
    if neighbor_entity_types:
        neighbor_descs = []
        for nt in neighbor_entity_types:
            if nt == root_entity:
                continue
            one_hop_entities.add(nt)
            nt_clean = _clean_entity_name(nt)
            nt_feats = metanode.get(nt, {}).get("feat", [])
            if nt_feats:
                cols = ", ".join(_clean_feat_name(f) for f in nt_feats[:6])
                if len(nt_feats) > 6:
                    cols += f", ... ({len(nt_feats)} total)"
                neighbor_descs.append(f"  - {nt_clean}: [{cols}]")
            else:
                neighbor_descs.append(f"  - {nt_clean}")
        if neighbor_descs:
            lines.append("")
            lines.append("Related tables and their key features:")
            lines.extend(neighbor_descs)

    # ── Section 4b: Second-hop entity descriptions ──
    # Discover entities connected to 1-hop neighbors via metaadj.
    # These are encoded in the GNN embedding but not yet described.
    two_hop_entities = set()
    for one_hop in one_hop_entities:
        oh_adj = metaadj.get(one_hop, {})
        for edge in oh_adj.get("in", []) + oh_adj.get("out", []):
            parts = edge.split(":")
            if len(parts) >= 3:
                src = parts[0].removeprefix("tail of ").removeprefix("head of ")
                dst = parts[2]
                for entity in (src, dst):
                    if entity != root_entity and entity not in one_hop_entities:
                        two_hop_entities.add(entity)

    if two_hop_entities:
        two_hop_descs = []
        for th in sorted(two_hop_entities):
            th_feats = metanode.get(th, {}).get("feat", [])
            if not th_feats:
                continue
            th_clean = _clean_entity_name(th)
            cols = ", ".join(_clean_feat_name(f) for f in th_feats[:6])
            if len(th_feats) > 6:
                cols += f", ... ({len(th_feats)} total)"
            two_hop_descs.append(f"  - {th_clean}: [{cols}]")
        if two_hop_descs:
            lines.append("")
            lines.append("Second-hop related tables:")
            lines.extend(two_hop_descs)

    # ── Section 5: Embedding explanation ──
    lines.append("")
    lines.append(
        "The following soft-token embeddings encode these features "
        "and relational neighborhood from a graph neural network. "
        "Use them alongside the schema description to make your prediction."
    )

    return "\n".join(lines) + "\n"
