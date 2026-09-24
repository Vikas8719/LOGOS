// ============================================================
//  LOGOS — Trainer.cpp
//
//  ⚠️  DEAD FILE — NOT COMPILED — NOT USED ⚠️
//
//  BUG 6: Trainer class koi bhi file include ya use nahi karti.
//    main.cpp apna training loop directly likhta hai.
//    CMakeLists.txt sirf src/main.cpp compile karta hai.
//    Trainer.cpp kabhi link nahi hoti.
//
//  BUG 6 (internal — agar yeh kabhi use hoti):
//    train_step() mein logit_grad compute hota hai lekin
//    grads[] mein KABHI COPY NAHI HOTA. Zero gradient
//    optimizer ko jaata hai — training silent no-op hai.
//
//    Broken code tha:
//      Tensor logit_grad(logits.shape, 0.0f);
//      // ... logit_grad fill karo ...
//      // grads mein COPY NAHI  ← BUG
//      std::vector<Tensor*> grad_ptrs;
//      for (auto& g : grads) grad_ptrs.push_back(&g);  // grads = all zeros!
//      optimizer.step(params, grad_ptrs);   // zero gradient step
//
//  WHY THIS FILE EXISTS (historical):
//    Phase 3 mein ek alag Trainer abstraction planned tha.
//    main.cpp ne training loop inline le liya — Trainer abandoned.
//    File git history ke liye rakhi gayi hai.
//
//  ACTUAL TRAINING CODE: src/main.cpp → run_training()
//
//  FUTURE: Agar Trainer class revive karni ho toh:
//    1. logit_grad ko grads[2] (lm_head grad) mein copy karo
//    2. dX compute karo (lm_head backprop)
//    3. grads[0] (embedding) aur grads[1] (pos_emb) fill karo
//    4. Transformer layer grad proxy compute karo (dX_norm)
//    5. main.cpp se training loop yahan move karo
//    6. CMakeLists.txt mein Trainer.cpp ko add karo
// ============================================================

// Intentionally empty — live training code in src/main.cpp → run_training()
